from statistics import median


def _values(records: list[dict], field: str) -> list:
    return [r[field] for r in records if r.get(field) is not None]


def accuracy(records: list[dict]) -> float:
    return sum(1 for r in records if r["correct"]) / len(records)


def mean_latency(records: list[dict]) -> float:
    values = _values(records, "latency_seconds")
    return sum(values) / len(values)


def median_latency(records: list[dict]) -> float:
    return median(_values(records, "latency_seconds"))


def mean_input_tokens(records: list[dict]) -> float:
    values = _values(records, "input_tokens")
    return sum(values) / len(values)


def mean_output_tokens(records: list[dict]) -> float:
    values = _values(records, "output_tokens")
    return sum(values) / len(values)


def mean_reasoning_tokens(records: list[dict]) -> float | None:
    raise NotImplementedError


def stability(records: list[dict]) -> float:
    """Por caso: parte de corridas que coinciden con la respuesta más frecuente; promedio entre casos.

    La respuesta es la categoría predicha; una salida inválida cuenta como una sola respuesta '<invalid>'.
    Las llamadas con error se excluyen.
    """
    by_case: dict[str, list[str]] = {}
    for r in records:
        if r.get("status", "ok") != "ok":
            continue
        answer = r["predicted"] if r["valid_schema"] else "<invalid>"
        by_case.setdefault(r["case_id"], []).append(answer)
    shares = [max(answers.count(a) for a in set(answers)) / len(answers) for answers in by_case.values()]
    return sum(shares) / len(shares)


def total_cost(records: list[dict]) -> float:
    return sum(_values(records, "cost_usd"))


def format_compliance(records: list[dict]) -> float:
    """Parte de respuestas con JSON válido y categoría dentro del enum (independiente de si es la correcta)."""
    return sum(1 for r in records if r["valid_schema"]) / len(records)


def parse_rate(records: list[dict]) -> float:
    return sum(1 for r in records if r["parse_ok"]) / len(records)
