import json

from src.config import CATEGORIES


def verify_prediction(raw_output: str, expected: dict) -> dict:
    expected_category = expected.get("category")
    result = {
        "parse_ok": False,
        "valid_schema": False,
        "correct": False,
        "predicted": None,
        "expected": expected_category,
    }

    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return result
    result["parse_ok"] = True

    if not isinstance(parsed, dict):
        return result

    category = parsed.get("category")
    if isinstance(category, str):
        result["predicted"] = category
    if category not in CATEGORIES:
        return result

    result["valid_schema"] = True
    result["correct"] = category == expected_category
    return result
