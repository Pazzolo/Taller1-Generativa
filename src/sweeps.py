from src.metrics import accuracy, mean_output_tokens, parse_rate, stability

DECODING_MODEL = "propietario_economico"
TOPK_MODEL = "open_weight_pequeno"

TEMPERATURES = (0.0, 0.3, 0.7, 1.0, 1.5)
TOP_PS = (0.5, 0.9, 1.0)
TOP_KS = (1, 5, 40)
TOPK_TEMPERATURE = 0.7
GREEDY_TEMPERATURE = 0.0
RUNS = 5
MAX_OUTPUT_TOKENS = 128

DECODING_EXPERIMENT = "temperature_top_p"
TOPK_EXPERIMENT = "top_k"

DECODING_COLUMNS = [
    "model", "temperature", "top_p", "calls", "errors", "accuracy", "stability", "mean_output_tokens", "parse_rate",
]
TOPK_COLUMNS = [
    "model", "top_k", "temperature", "calls", "errors", "accuracy", "stability", "mean_output_tokens", "parse_rate",
    "greedy_match",
]


def decoding_cells() -> list[dict]:
    return [{"temperature": t, "top_p": p, "top_k": None} for t in TEMPERATURES for p in TOP_PS]


def topk_cells() -> list[dict]:
    baseline = {"temperature": GREEDY_TEMPERATURE, "top_p": None, "top_k": None}
    return [baseline] + [{"temperature": TOPK_TEMPERATURE, "top_p": None, "top_k": k} for k in TOP_KS]


def cell_rows(rows: list[dict], model: str, experiment: str, cell: dict) -> list[dict]:
    """Filas de una celda; si una llamada se repitió (caso, corrida) gana la última."""
    latest = {}
    for r in rows:
        if (
            r.get("part") == "2b"
            and r["model_id"] == model
            and r["experiment"] == experiment
            and all(r.get(k) == v for k, v in cell.items())
        ):
            latest[(r["case_id"], r["run"])] = r
    return list(latest.values())


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    return {
        "calls": len(records),
        "errors": len(records) - len(ok),
        "accuracy": accuracy(records),
        "stability": stability(ok) if ok else None,
        "mean_output_tokens": mean_output_tokens(ok) if ok else None,
        "parse_rate": parse_rate(records),
    }


def decoding_table(rows: list[dict], model: str = DECODING_MODEL) -> list[dict]:
    table = []
    for cell in decoding_cells():
        records = cell_rows(rows, model, DECODING_EXPERIMENT, cell)
        if records:
            table.append({"model": model, "temperature": cell["temperature"], "top_p": cell["top_p"], **summarize(records)})
    return table


def _modal_output_by_case(records: list[dict]) -> dict:
    by_case: dict[str, list[str]] = {}
    for r in records:
        if r["status"] == "ok":
            by_case.setdefault(r["case_id"], []).append(r["raw_output"].strip())
    return {case: max(set(outs), key=outs.count) for case, outs in by_case.items()}


def greedy_match(topk1_records: list[dict], greedy_records: list[dict]) -> float | None:
    """Parte de casos donde TODAS las corridas con top_k=1 producen exactamente la salida modal del baseline greedy."""
    reference = _modal_output_by_case(greedy_records)
    by_case: dict[str, list[str]] = {}
    for r in topk1_records:
        if r["status"] == "ok" and r["case_id"] in reference:
            by_case.setdefault(r["case_id"], []).append(r["raw_output"].strip())
    if not by_case:
        return None
    return sum(all(o == reference[c] for o in outs) for c, outs in by_case.items()) / len(by_case)


def topk_table(rows: list[dict], model: str = TOPK_MODEL) -> list[dict]:
    baseline_cell, *cells = topk_cells()
    greedy = cell_rows(rows, model, TOPK_EXPERIMENT, baseline_cell)
    table = []
    for cell in [baseline_cell, *cells]:
        records = greedy if cell is baseline_cell else cell_rows(rows, model, TOPK_EXPERIMENT, cell)
        if not records:
            continue
        match = greedy_match(records, greedy) if cell["top_k"] == 1 else None
        table.append({"model": model, "top_k": cell["top_k"], "temperature": cell["temperature"], **summarize(records), "greedy_match": match})
    return table
