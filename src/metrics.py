def accuracy(records: list[dict]) -> float:
    return sum(1 for r in records if r["correct"]) / len(records)


def mean_latency(records: list[dict]) -> float:
    values = [r["latency_seconds"] for r in records if r.get("latency_seconds") is not None]
    return sum(values) / len(values)


def median_latency(records: list[dict]) -> float:
    raise NotImplementedError


def mean_output_tokens(records: list[dict]) -> float:
    raise NotImplementedError


def mean_reasoning_tokens(records: list[dict]) -> float | None:
    raise NotImplementedError


def stability(records: list[dict]) -> float:
    raise NotImplementedError


def total_cost(records: list[dict]) -> float:
    return sum(r["cost_usd"] for r in records if r.get("cost_usd") is not None)


def parse_rate(records: list[dict]) -> float:
    raise NotImplementedError
