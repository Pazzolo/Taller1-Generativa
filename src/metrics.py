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
    raise NotImplementedError


def total_cost(records: list[dict]) -> float:
    return sum(_values(records, "cost_usd"))


def parse_rate(records: list[dict]) -> float:
    return sum(1 for r in records if r["parse_ok"]) / len(records)
