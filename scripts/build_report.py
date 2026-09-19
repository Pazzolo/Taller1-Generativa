"""Genera report/report.md a partir de report/report_template.md.

Cada número del informe sale de outputs/tables/*.csv, outputs/raw/part0*.json y results.jsonl:
la plantilla solo contiene texto y marcas [[clave]] / [[tabla:nombre]].
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CASES_PATH, CONTAMINATED_PATH, RAW_DIR, RESULTS_PATH, ROOT, TABLES_DIR
from src.pricing import PRICES
from src.reasoning import latest_rows
from src.results import read_results
from src.schemas import load_cases

TEMPLATE = ROOT / "report" / "report_template.md"
NOMBRES = {
    "high_confidence": "alta confianza", "low_confidence": "baja confianza",
    "rejected": "rechazado", "accepted_and_acts": "acepta y actúa", "accepted_and_does_not_act": "acepta y no actúa", "inconclusive": "no concluyente",
    "distraction": "distracción", "misleading_frame": "marco engañoso", "spurious_correlation": "correlación espuria",
}


def nombre(value: str) -> str:
    return NOMBRES.get(value, value)


OUTPUT = ROOT / "report" / "report.md"
Q1_WORDS = (100, 150)


def read_csv(name: str) -> list[dict]:
    with open(TABLES_DIR / f"{name}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(name: str):
    with open(RAW_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def num(value, decimals=2) -> str:
    return "—" if value in ("", None) else f"{float(value):.{decimals}f}"


def usd(value, decimals=4) -> str:
    return "—" if value in ("", None) else f"${float(value):.{decimals}f}"


def md_table(rows: list[dict], spec: list[tuple]) -> str:
    """spec: (encabezado, clave o función, formateador opcional)."""
    lines = ["| " + " | ".join(h for h, *_ in spec) + " |", "|" + "|".join("---" for _ in spec) + "|"]
    for row in rows:
        cells = []
        for _, key, *fmt in spec:
            value = key(row) if callable(key) else row[key]
            cells.append(fmt[0](value) if fmt else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def literal_message(text: str) -> str:
    match = re.search(r"'message': (?:\"([^\"]+)\"|'([^']+)')", text or "")
    return (match.group(1) or match.group(2)) if match else (text or "")


def build_values() -> tuple[dict, dict]:
    v, tables = {}, {}
    rows = read_results(RESULTS_PATH)
    real = [r for r in rows if r["model_id"] != "mock"]

    # --- totales
    v["n_rows"] = f"{len(real):,}".replace(",", " ")
    v["n_errors"] = str(sum(r["status"] != "ok" for r in real))
    v["total_cost"] = usd(sum(r.get("cost_usd") or 0 for r in real), 3)
    by_part = {}
    for r in real:
        entry = by_part.setdefault(r["part"], [0, 0, 0.0])
        entry[0] += 1
        entry[1] += r["status"] != "ok"
        entry[2] += r.get("cost_usd") or 0
    labels = {"1": "Parte 1", "2a": "Parte 2.a", "2b": "Parte 2.b", "3": "Parte 3", "4a": "Parte 4.a", "4b": "Parte 4.b"}
    tables["costos"] = md_table(
        [{"parte": labels[p], "llamadas": n, "errores": e, "costo": c} for p, (n, e, c) in sorted(by_part.items())],
        [("Parte", "parte"), ("Llamadas", "llamadas"), ("Errores (rechazos esperados)", "errores"), ("Costo (USD)", "costo", lambda x: usd(x, 4))],
    )

    # --- dataset
    official, debug = load_cases(split="official"), load_cases(split="debug")
    contaminated = load_cases(CONTAMINATED_PATH, split="contaminated")
    counts = Counter(c.expected.category for c in official)
    v.update(n_official=str(len(official)), n_debug=str(len(debug)), n_contaminated=str(len(contaminated)),
             n_billing=str(counts["billing"]), n_technical=str(counts["technical"]), n_account=str(counts["account"]))

    # --- Parte 0
    scan = read_csv("part0_prefix_scan")
    france = next(s for s in scan if s["prefix"].startswith("The capital of France"))
    chosen = next(s for s in scan if s["chosen_as"] == "high_confidence")
    low = next(s for s in scan if s["chosen_as"] == "low_confidence")
    v.update(france_entropy=num(france["entropy_bits_t1"]), france_top=france["top_token"].strip("'").strip(), france_top_p=num(france["top_prob"], 3),
             chosen_prefix=chosen["prefix"], chosen_entropy=num(chosen["entropy_bits_t1"]), chosen_top=chosen["top_token"].strip("'").strip(),
             chosen_top_p=num(chosen["top_prob"], 3), low_entropy=num(low["entropy_bits_t1"]), n_candidates=str(len(scan) - 1))
    tables["prefijos"] = md_table(scan, [("Prefijo", "prefix"), ("Entropía a T=1 (bits)", "entropy_bits_t1", lambda x: num(x)),
                                          ("Token más probable", "top_token"), ("p", "top_prob", lambda x: num(x, 3)), ("Elegido como", "chosen_as", nombre)])
    temp = read_csv("part0_temperature")
    tables["part0a"] = md_table(temp, [("Prefijo", "prefix"), ("T", "temperature", lambda x: num(x, 1)),
                                       ("Entropía (bits)", "entropy_bits", lambda x: num(x)), ("Núcleo top-p 0.9", "nucleus_size_p90")])
    b = read_json("part0b.json")
    t1, t2 = b["test1_do_sample_false_ignores_temperature"], b["test2_top_k_1"]
    panels = {p["prefix_id"]: p for p in b["test3_top_k_vs_top_p"]["panels"]}
    loop = b["test4_degeneration"]["raw_output"]
    v.update(
        t1_identical="idéntica" if t1["identical"] else "distinta",
        t1_warning=t1["0.2"]["warnings"][0] if t1["0.2"]["warnings"] else "(la librería no emitió aviso)",
        t2_identical="idénticas entre sí" if t2["all_identical_to_each_other"] else "distintas entre sí",
        t2_greedy="iguales a greedy" if t2["all_equal_to_greedy"] else "no siempre iguales a greedy",
        t2_seeds=f"{t2['runs'][0]['seed']}–{t2['runs'][-1]['seed']}",
        k_high=str(panels["high_confidence"]["top_k_size"]), p_high=str(panels["high_confidence"]["top_p_size"]),
        k_low=str(panels["low_confidence"]["top_k_size"]), p_low=str(panels["low_confidence"]["top_p_size"]),
        loop_repeats=str(loop.count("The customer support ticket about an unexpected problem")), loop_tokens=str(b["test4_degeneration"]["max_new_tokens"]),
        gen_prefix=b["prefix"], greedy_head=b["greedy_reference"][:40].replace("\n", " ").strip(),
    )
    c = read_json("part0c.json")
    ticket_of = lambda r: r["prompt"].split("Ticket:\n")[-1]
    repeats = sum(ticket_of(r) in r["raw_output"] for r in c["records"])
    example = lambda cid: " ".join(next(r for r in c["records"] if r["case_id"] == cid)["raw_output"].split())[:170]
    v.update(c_parse=str(c["parse_ok"]), c_valid=str(c["valid_schema"]), c_correct=str(c["correct"]), c_cases=str(c["cases"]),
             c_repeats=str(repeats), c_ex1=example("case_01"), c_ex9=example("case_09"))

    # --- Parte 1
    part1 = read_csv("part1")
    p1 = {r["model"]: r for r in part1}
    tables["part1"] = md_table(part1, [
        ("Modelo", "model_name"), ("Exactitud", "accuracy", num), ("Latencia media (s)", "latency_mean", num), ("Latencia p50 (s)", "latency_p50", num),
        ("Tokens entrada", "input_tokens_mean", lambda x: num(x, 1)), ("Tokens salida", "output_tokens_mean", lambda x: num(x, 1)),
        ("Costo total", "cost_usd_total", lambda x: usd(x, 5)), ("Parse rate", "parse_rate", num)])
    first_local = next(r for r in rows if r["part"] == "1" and r["model_id"] == "open_weight_pequeno" and r["status"] == "ok")
    v.update(p1_min_correct=str(min(round(float(r["accuracy"]) * int(r["cases"])) for r in part1)),
             p1_min_parse=num(min(float(r["parse_rate"]) for r in part1)), p1_local_first=num(first_local["latency_seconds"]),
             cost_ratio_big_small=f"{float(p1['propietario_grande']['cost_usd_total']) / float(p1['propietario_economico']['cost_usd_total']):.0f}",
             p1_big_lat=num(p1["propietario_grande"]["latency_mean"]), p1_small_lat=num(p1["propietario_economico"]["latency_mean"]),
             p1_local_lat=num(p1["open_weight_pequeno"]["latency_mean"]), p1_big_out=num(p1["propietario_grande"]["output_tokens_mean"], 1))
    tables["modelos"] = md_table(
        [{"clave": k, **PRICES[k]} for k in ("propietario_grande", "propietario_economico", "open_weight_pequeno", "openai_razonamiento")],
        [("Clave", "clave"), ("Proveedor", "provider"), ("Modelo", "model_id"), ("USD/1M entrada", "input_per_million", lambda x: f"{x:g}"),
         ("USD/1M salida", "output_per_million", lambda x: f"{x:g}"), ("Precio verificado", "verified_at")])

    # --- Parte 2.a
    part2a = read_csv("part2a")
    tables["part2a"] = md_table(part2a, [
        ("Modelo", "model"), ("Parámetro", "parameter"), ("Declarado", "declared"), ("Observado", "observed_state", nombre),
        ("HTTP", "http_status", lambda x: x or "200"), ("Salidas distintas (bajo / alto)", lambda r: f"{r['distinct_low']} / {r['distinct_high']}"),
        ("Verificado", "verified_at")])
    seen, items = set(), []
    for r in part2a:
        if r["error_message"] and (r["model"], literal_message(r["error_message"])) not in seen:
            seen.add((r["model"], literal_message(r["error_message"])))
            items.append(f"- `{r['model']}`, `{r['parameter']}` (HTTP {r['http_status']}): «{literal_message(r['error_message'])}»")
    tables["errores_2a"] = "\n".join(items)
    v["n_2a_rejected"] = str(sum(r["observed_state"] == "rejected" for r in part2a))
    v["n_2a_acts"] = str(sum(r["observed_state"] == "accepted_and_acts" for r in part2a))
    v["n_2a_cells"] = str(len(part2a))

    # --- Parte 2.b
    dec = read_csv("part2b_decoding")
    grid = {(r["temperature"], r["top_p"]): r for r in dec}
    temps = sorted({r["temperature"] for r in dec}, key=float)
    tops = sorted({r["top_p"] for r in dec}, key=float)
    lines = ["| Temperatura \\ top-p | " + " | ".join(num(t, 1) for t in tops) + " |", "|---|" + "---|" * len(tops)]
    for t in temps:
        lines.append(f"| {num(t, 1)} | " + " | ".join(f"{num(grid[(t, p)]['accuracy'])} / {num(grid[(t, p)]['stability'])}" for p in tops) + " |")
    tables["part2b_grid"] = "\n".join(lines)
    v.update(n_dec_cells=str(len(dec)), n_2b_calls=str(sum(int(r["calls"]) for r in dec) + sum(int(r["calls"]) for r in read_csv("part2b_topk"))), n_dec_calls=str(sum(int(r["calls"]) for r in dec)),
             dec_min_acc=num(min(float(r["accuracy"]) for r in dec)), dec_min_stab=num(min(float(r["stability"]) for r in dec)),
             dec_errors=str(sum(int(r["errors"]) for r in dec)))
    odd = [r for r in dec if abs(float(r["mean_output_tokens"]) - 6.0) > 1e-9]
    v["dec_odd"] = f"temperatura {num(odd[0]['temperature'], 1)} y top-p {num(odd[0]['top_p'], 1)} ({odd[0]['mean_output_tokens']} tokens de salida en media)" if odd else "ninguna"
    topk = read_csv("part2b_topk")
    tables["part2b_topk"] = md_table(topk, [
        ("top-k", "top_k", lambda x: x or "— (greedy)"), ("Temperatura", "temperature", lambda x: num(x, 1)), ("Llamadas", "calls"),
        ("Exactitud", "accuracy", num), ("Estabilidad", "stability", num), ("Coincide con greedy", "greedy_match", lambda x: num(x) if x else "—")])
    k40 = next(r for r in topk if r["top_k"] == "40")
    v.update(k40_acc=num(k40["accuracy"]), k1_match=num(next(r for r in topk if r["top_k"] == "1")["greedy_match"]),
             n_topk_calls=str(sum(int(r["calls"]) for r in topk)))

    # --- Parte 3
    part3 = read_csv("part3")
    p3 = {r["prompt_variant"]: r for r in part3}
    base_cost = float(p3["zero_shot"]["cost_usd_total"])
    tables["part3"] = md_table(part3, [
        ("Variante", "prompt_variant"), ("Exactitud", "accuracy", num), ("Formato válido", "format_compliance", num),
        ("Tokens entrada", "input_tokens_mean", lambda x: num(x, 1)), ("Tokens salida", "output_tokens_mean", lambda x: num(x, 1)),
        ("Costo entrada", "cost_input_usd", lambda x: usd(x, 5)), ("Costo salida", "cost_output_usd", lambda x: usd(x, 5)),
        ("Costo total", "cost_usd_total", lambda x: usd(x, 5)), ("Frente a zero-shot", lambda r: f"{float(r['cost_usd_total']) / base_cost:.1f}×")])
    v.update(few_ratio=f"{float(p3['few_shot']['cost_usd_total']) / base_cost:.1f}", cot_ratio=f"{float(p3['cot']['cost_usd_total']) / base_cost:.1f}",
             struct_ratio=f"{float(p3['structured']['cost_usd_total']) / base_cost:.1f}", cot_out=num(p3["cot"]["output_tokens_mean"], 1),
             zero_out=num(p3["zero_shot"]["output_tokens_mean"], 1), cot_out_mult=f"{float(p3['cot']['output_tokens_mean']) / float(p3['zero_shot']['output_tokens_mean']):.0f}",
             zero_in=num(p3["zero_shot"]["input_tokens_mean"], 1), few_in=num(p3["few_shot"]["input_tokens_mean"], 1),
             struct_in=num(p3["structured"]["input_tokens_mean"], 1), struct_out=num(p3["structured"]["output_tokens_mean"], 1),
             out_price=f"{PRICES['propietario_economico']['output_per_million']:g}", in_price=f"{PRICES['propietario_economico']['input_per_million']:g}",
             p3_calls=str(sum(int(r["calls"]) for r in part3)))

    # --- Parte 4.a
    part4a = read_csv("part4a")
    ok4a = [r for r in part4a if r["accuracy"]]
    tables["part4a"] = md_table(part4a, [
        ("Esfuerzo", "effort"), ("Llamadas", "calls"), ("Exactitud", "accuracy", lambda x: num(x) if x else "—"),
        ("Razonamiento (media)", "reasoning_tokens_mean", lambda x: num(x) if x else "—"), ("Salida total (media)", "total_output_tokens_mean", lambda x: num(x, 1) if x else "—"),
        ("Costo por llamada", "cost_usd_per_call", lambda x: usd(x, 7) if x else "—")])
    ctrl = read_csv("part4a_control")
    tables["part4a_control"] = md_table(ctrl, [("Esfuerzo", "effort"), ("Corridas", "calls"), ("Razonamiento (media)", "reasoning_tokens_mean", lambda x: num(x, 1)),
                                                ("Salida total (media)", "total_output_tokens_mean", lambda x: num(x, 1)), ("Respuesta correcta", "answer_correct_rate", num)])
    sweep_rows = [r for lvl in ("none", "low", "medium", "high", "xhigh") for r in latest_rows(rows, "openai_razonamiento", "reasoning_effort", lvl) if r["status"] == "ok"]
    reasoned = [r for r in sweep_rows if r["reasoning_tokens"]]
    cheapest, dearest = float(ok4a[0]["cost_usd_per_call"]), float(ok4a[-1]["cost_usd_per_call"])
    max_row = next(r for r in part4a if r["effort"] == "max")
    v.update(n_sweep=str(len(sweep_rows)), n_reasoned=str(len(reasoned)), reasoned_cases=" y ".join(f"`{c}`" for c in sorted({r["case_id"] for r in reasoned})),
             xhigh_extra=f"{100 * (dearest / cheapest - 1):.0f}", max_message=literal_message(max_row["error_message"]),
             acc_min=num(min(float(r["accuracy"]) for r in ok4a)), acc_max=num(max(float(r["accuracy"]) for r in ok4a)),
             ctrl_none=num(ctrl[0]["reasoning_tokens_mean"], 0), ctrl_xhigh=num(ctrl[-1]["reasoning_tokens_mean"], 0),
             ctrl_series=" / ".join(num(r["reasoning_tokens_mean"], 0) for r in ctrl[1:]), n_levels_4a=str(len(ok4a)))

    # --- Parte 4.b
    part4b = read_csv("part4b")
    tables["part4b"] = md_table(part4b, [
        ("Caso", "case_id"), ("Trampa", "trap", nombre), ("Esfuerzo", "effort"), ("Corridas", "runs"), ("Aciertos", "correct_rate", num),
        ("Razonamiento (media)", "reasoning_tokens_mean", lambda x: num(x, 1)), ("Razonamiento (máx.)", "reasoning_tokens_max")])
    calls4b = sum(int(r["runs"]) for r in part4b)
    right4b = sum(float(r["correct_rate"]) * (int(r["runs"]) - int(r["errors"])) for r in part4b)
    v.update(n_4b_calls=str(calls4b), n_4b_right=f"{right4b:.0f}", max_reasoning_4b=str(max(int(r["reasoning_tokens_max"]) for r in part4b)))
    return v, tables


def render(template: str, v: dict, tables: dict) -> str:
    def replace(match):
        key = match.group(1).strip()
        if key.startswith("tabla:"):
            return tables[key[len("tabla:"):]]
        return v[key]

    return re.sub(r"\[\[([^\]]+)\]\]", replace, template)


def q1_word_count(text: str) -> int:
    body = re.search(r"<!--Q1-->(.*?)<!--/Q1-->", text, re.S).group(1)
    return len(body.split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera report/report.md desde la plantilla.")
    parser.add_argument("--author", help="nombre del estudiante (reemplaza [nombre] en la portada)")
    args = parser.parse_args()

    values, tables = build_values()
    text = render(TEMPLATE.read_text(encoding="utf-8"), values, tables)
    leftovers = re.findall(r"\[\[[^\]]+\]\]", text)
    assert not leftovers, f"marcas sin resolver: {leftovers}"
    words = q1_word_count(text)
    assert Q1_WORDS[0] <= words <= Q1_WORDS[1], f"la respuesta 1 tiene {words} palabras (límite {Q1_WORDS})"
    text = text.replace("<!--Q1-->\n", "").replace("\n<!--/Q1-->", "")
    if args.author:
        text = text.replace("[nombre]", args.author)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"report/report.md generado: {len(text.split())} palabras en total; respuesta 1: {words} palabras")


if __name__ == "__main__":
    main()
