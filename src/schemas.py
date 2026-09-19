import json
from pathlib import Path

from pydantic import BaseModel

from src.config import CASES_PATH, CATEGORIES, Category


class Expected(BaseModel):
    category: Category


class Case(BaseModel):
    id: str
    ticket: str
    expected: Expected


CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
    },
    "required": ["category"],
    "additionalProperties": False,
}


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    with open(path, encoding="utf-8") as f:
        return [Case(**item) for item in json.load(f)]
