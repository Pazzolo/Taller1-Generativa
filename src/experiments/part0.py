import argparse
import csv
import json

import numpy as np

from src.config import PLOTS_DIR, RAW_DIR, RESULTS_PATH, SEED, TABLES_DIR
from src.distribution import (
    entropy_bits,
    nucleus_size,
    ranked,
    temperature_probs,
    top_k_indices,
    top_p_indices,
)
from src.models.transformers_runner import TransformersRunner
from src.prompts import build_base_prompt
from src.results import call_key, completed_keys, read_results
from src.runner import run_case
from src.schemas import Case, Expected, load_cases

MODEL_NAME = "openai-community/gpt2"
MODEL_KEY = "gpt2_base"  # fila de src/pricing.py: costo 0, modelo local
# El ejemplo del plan es el primer candidato; se elige el de menor entropía (T=1) medida, no supuesta.
HIGH_CONFIDENCE_CANDIDATES = (
    "The capital of France is",
    "The quick brown fox jumps over the lazy",
    "Once upon a",
    "Thank you very",
    "The United States of",
    "Happy birthday to",
    "1, 2, 3, 4, 5, 6,",
)
LOW_CONFIDENCE_PREFIX = "A customer support ticket about an unexpected"
TEMPERATURES = (0.1, 0.7, 1.0, 1.5, 2.0)
TOP_P = 0.9
TOP_K = 5
TOP_N = 15
NEW_TOKENS = 40
DEGENERATION_TOKENS = 100
PART0C_TOKENS = 60


def load_gpt2():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
    return tokenizer, model


def next_token_logits(tokenizer, model, prefix: str) -> np.ndarray:
    import torch

    inputs = tokenizer(prefix, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    return logits.double().numpy()


def token_text(tokenizer, index: int) -> str:
    return tokenizer.decode([int(index)])


def most_confident(entropies: dict[str, float]) -> str:
    return min(entropies, key=entropies.get)


def scan_prefixes(tokenizer, model) -> dict:
    """Mide la entropía a T=1 de cada candidato y elige el prefijo de alta confianza."""
    scan = []
    for prefix in (*HIGH_CONFIDENCE_CANDIDATES, LOW_CONFIDENCE_PREFIX):
        probs = temperature_probs(next_token_logits(tokenizer, model, prefix), 1.0)
        top = int(ranked(probs)[0])
        scan.append(
            {"prefix": prefix, "entropy_bits_t1": entropy_bits(probs), "top_token": token_text(tokenizer, top), "top_prob": float(probs[top])}
        )
    high = most_confident({s["prefix"]: s["entropy_bits_t1"] for s in scan if s["prefix"] in HIGH_CONFIDENCE_CANDIDATES})
    return {"scan": scan, "prefixes": {"high_confidence": high, "low_confidence": LOW_CONFIDENCE_PREFIX}}


def part0a(tokenizer, model, prefixes: dict) -> list[dict]:
    records = []
    for name, prefix in prefixes.items():
        logits = next_token_logits(tokenizer, model, prefix)
        for temperature in TEMPERATURES:
            probs = temperature_probs(logits, temperature)
            order = ranked(probs)[:TOP_N]
            records.append(
                {
                    "prefix_id": name,
                    "prefix": prefix,
                    "temperature": temperature,
                    "entropy_bits": entropy_bits(probs),
                    "nucleus_size_p90": nucleus_size(probs, TOP_P),
                    "top_tokens": [token_text(tokenizer, i) for i in order],
                    "top_probs": [float(probs[i]) for i in order],
                }
            )
    return records


def part0b_test3(tokenizer, model, prefixes: dict) -> dict:
    """top-k vs top-p sobre la misma distribución (T = 1): qué tokens sobreviven a cada corte."""
    panels = []
    for name, prefix in prefixes.items():
        probs = temperature_probs(next_token_logits(tokenizer, model, prefix), 1.0)
        keep_k, keep_p = top_k_indices(probs, TOP_K), top_p_indices(probs, TOP_P)
        order = ranked(probs)[:TOP_N]

        def membership(i):
            return "both" if i in keep_k and i in keep_p else "top_k_only" if i in keep_k else "top_p_only" if i in keep_p else "neither"

        panels.append(
            {
                "prefix_id": name,
                "prefix": prefix,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "top_k_size": len(keep_k),
                "top_p_size": len(keep_p),
                "sets_differ": keep_k != keep_p,
                "tokens": [token_text(tokenizer, i) for i in order],
                "probs": [float(probs[i]) for i in order],
                "membership": [membership(int(i)) for i in order],
                "top_k_tokens": [token_text(tokenizer, i) for i in sorted(keep_k, key=lambda i: -probs[i])],
                "top_p_tokens": [token_text(tokenizer, i) for i in sorted(keep_p, key=lambda i: -probs[i])],
            }
        )
    return {"panels": panels}


def prefix_case(prefix: str) -> Case:
    """Caso auxiliar: el prefijo hace de «ticket»; los campos de verificación de esas filas no significan nada."""
    return Case(id="prefix_low_confidence", ticket=prefix, expected=Expected(category="billing"), split="debug")


def generation_plan(base: TransformersRunner, prefixes: dict) -> list[tuple]:
    """Todas las generaciones locales de las Partes 0.b y 0.c: (experimento, runner, caso, corrida, parámetros, prompt)."""
    case = prefix_case(prefixes["low_confidence"])
    greedy = base.clone(do_sample=False, seed=None, max_new_tokens=NEW_TOKENS)
    plan = [("greedy_reference", greedy, case, 1, {}, "raw_prefix")]
    plan += [("greedy_ignores_temperature", greedy, case, 1, {"temperature": t}, "raw_prefix") for t in (0.2, 1.5)]
    plan += [
        ("top_k_1", base.clone(do_sample=True, seed=SEED + i, max_new_tokens=NEW_TOKENS), case, i + 1,
         {"temperature": 1.0, "top_p": 1.0, "top_k": 1}, "raw_prefix")
        for i in range(5)
    ]
    plan.append(("degeneration", base.clone(do_sample=False, seed=None, max_new_tokens=DEGENERATION_TOKENS), case, 1, {}, "raw_prefix"))
    classify = base.clone(do_sample=False, seed=None, max_new_tokens=PART0C_TOKENS)
    plan += [("base_model_classification", classify, c, 1, {}, "base") for c in load_cases()]
    return plan


def plan_key(experiment, case, run, params, prompt_variant) -> tuple:
    return call_key({
        "model_id": MODEL_KEY, "experiment": experiment, "case_id": case.id, "run": run,
        "temperature": params.get("temperature"), "top_p": params.get("top_p"), "top_k": params.get("top_k"),
        "effort": None, "prompt_variant": prompt_variant,
    })


def run_generations(base: TransformersRunner, prefixes: dict, only: str | None) -> None:
    """Cada generación es una fila de results.jsonl (costo 0). Las ya registradas se omiten: la Parte 0 es determinista."""
    wanted = {"b": {"greedy_reference", "greedy_ignores_temperature", "top_k_1", "degeneration"}, "c": {"base_model_classification"}}
    experiments = wanted["b"] | wanted["c"] if only is None else wanted[only]
    done = completed_keys(read_results(RESULTS_PATH))
    plan = [call for call in generation_plan(base, prefixes) if call[0] in experiments]
    pending = [call for call in plan if plan_key(call[0], call[2], call[3], call[4], call[5]) not in done]
    print(f"generaciones locales: {len(plan)} planificadas, {len(plan) - len(pending)} ya registradas, {len(pending)} por correr")
    for experiment, runner, case, run, params, prompt_variant in pending:
        run_case(runner, MODEL_KEY, case, part="0", experiment=experiment, run=run, prompt_variant=prompt_variant, **params)


def experiment_rows(rows: list[dict], experiment: str) -> list[dict]:
    """Filas de un experimento de la Parte 0; si una llamada se repitió (caso, corrida) gana la última."""
    latest = {}
    for r in rows:
        if r.get("part") == "0" and r["model_id"] == MODEL_KEY and r["experiment"] == experiment and r["status"] == "ok":
            latest[(r["case_id"], r["run"], r["temperature"], r["top_p"], r["top_k"])] = r
    return list(latest.values())


def part0b(tokenizer, model, prefixes: dict) -> dict:
    """Arma part0b.json a partir de las filas de results.jsonl (las generaciones) y de los logits (test 3)."""
    rows = read_results(RESULTS_PATH)
    greedy = experiment_rows(rows, "greedy_reference")[0]["raw_output"]

    test1 = {str(r["temperature"]): {"text": r["raw_output"], "warnings": r.get("notices", [])}
             for r in sorted(experiment_rows(rows, "greedy_ignores_temperature"), key=lambda r: r["temperature"])}
    test1["identical"] = test1["0.2"]["text"] == test1["1.5"]["text"] == greedy

    runs = [{"seed": SEED + r["run"] - 1, "text": r["raw_output"], "equals_greedy": r["raw_output"] == greedy}
            for r in sorted(experiment_rows(rows, "top_k_1"), key=lambda r: r["run"])]
    test2 = {
        "runs": runs,
        "all_identical_to_each_other": len({r["text"] for r in runs}) == 1,
        "all_equal_to_greedy": all(r["equals_greedy"] for r in runs),
    }
    degeneration = experiment_rows(rows, "degeneration")[0]["raw_output"]
    return {
        "model": MODEL_NAME,
        "prefix": prefixes["low_confidence"],
        "max_new_tokens": NEW_TOKENS,
        "seed_base": SEED,
        "greedy_reference": greedy,
        "test1_do_sample_false_ignores_temperature": test1,
        "test2_top_k_1": test2,
        "test3_top_k_vs_top_p": part0b_test3(tokenizer, model, prefixes),
        "test4_degeneration": {"max_new_tokens": DEGENERATION_TOKENS, "raw_output": degeneration},
    }


def part0c() -> dict:
    by_case = {r["case_id"]: r for r in experiment_rows(read_results(RESULTS_PATH), "base_model_classification")}
    records = []
    for case in load_cases():
        r = by_case[case.id]
        records.append({"case_id": case.id, "prompt": build_base_prompt(case.ticket), "raw_output": r["raw_output"],
                        "parse_ok": r["parse_ok"], "valid_schema": r["valid_schema"], "correct": r["correct"],
                        "predicted": r["predicted"], "expected": r["expected"]})
    return {
        "model": MODEL_NAME,
        "note": "GPT-2 base, decodificación greedy, mismo prompt que la Parte 1 (build_base_prompt).",
        "max_new_tokens": PART0C_TOKENS,
        "cases": len(records),
        "parse_ok": sum(r["parse_ok"] for r in records),
        "valid_schema": sum(r["valid_schema"] for r in records),
        "correct": sum(r["correct"] for r in records),
        "records": records,
    }


def write_json(data, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_part0a_outputs(chosen: dict, records: list[dict]) -> None:
    from src.plots_part0 import plot_entropy, plot_temperature

    write_json({"prefix_scan": chosen["scan"], "prefixes": chosen["prefixes"], "records": records}, RAW_DIR / "part0a.json")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / "part0_prefix_scan.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix", "entropy_bits_t1", "top_token", "top_prob", "chosen_as"])
        chosen_as = {v: k for k, v in chosen["prefixes"].items()}
        for s in chosen["scan"]:
            writer.writerow([s["prefix"], f"{s['entropy_bits_t1']:.4f}", repr(s["top_token"]), f"{s['top_prob']:.4f}", chosen_as.get(s["prefix"], "")])
    with open(TABLES_DIR / "part0_temperature.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix", "temperature", "entropy_bits", "nucleus_size_p90"])
        for r in records:
            writer.writerow([r["prefix"], r["temperature"], f"{r['entropy_bits']:.4f}", r["nucleus_size_p90"]])

    for prefix_id in chosen["prefixes"]:
        entries = [r for r in records if r["prefix_id"] == prefix_id]
        plot_temperature(entries[0]["prefix"], entries, PLOTS_DIR / f"part0_temperature_{prefix_id}.png")
    series = {
        f"«{chosen['prefixes'][pid]}»": [(r["temperature"], r["entropy_bits"]) for r in records if r["prefix_id"] == pid]
        for pid in chosen["prefixes"]
    }
    plot_entropy(series, PLOTS_DIR / "part0_entropy_vs_temperature.png")


def write_part0b_outputs(result: dict) -> None:
    from src.plots_part0 import plot_topk_vs_topp

    write_json(result, RAW_DIR / "part0b.json")
    panels = result["test3_top_k_vs_top_p"]["panels"]
    plot_topk_vs_topp(panels, TOP_K, TOP_P, PLOTS_DIR / "part0_topk_vs_topp.png")
    with open(TABLES_DIR / "part0_topk_vs_topp.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix", "rank", "token", "probability", "survives"])
        for panel in panels:
            for rank, (token, prob, member) in enumerate(zip(panel["tokens"], panel["probs"], panel["membership"]), 1):
                writer.writerow([panel["prefix"], rank, repr(token), f"{prob:.6f}", member])


def main() -> None:
    parser = argparse.ArgumentParser(description="Parte 0: GPT-2 base local (sin API).")
    parser.add_argument("--only", choices=["a", "b", "c"], help="ejecutar solo una subparte")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_gpt2()
    chosen = scan_prefixes(tokenizer, model)
    prefixes = chosen["prefixes"]
    print(f"prefijo de alta confianza elegido: «{prefixes['high_confidence']}»")
    if args.only in (None, "a"):
        write_part0a_outputs(chosen, part0a(tokenizer, model, prefixes))
        print("Parte 0.a: outputs/raw/part0a.json, tablas y gráficas")
    if args.only in (None, "b", "c"):
        run_generations(TransformersRunner(tokenizer, model), prefixes, args.only)
    if args.only in (None, "b"):
        write_part0b_outputs(part0b(tokenizer, model, prefixes))
        print("Parte 0.b: outputs/raw/part0b.json y gráfica top-k vs top-p")
    if args.only in (None, "c"):
        result = part0c()
        write_json(result, RAW_DIR / "part0c.json")
        print(f"Parte 0.c: outputs/raw/part0c.json  parse_ok={result['parse_ok']}/{result['cases']}  correct={result['correct']}/{result['cases']}")


if __name__ == "__main__":
    main()
