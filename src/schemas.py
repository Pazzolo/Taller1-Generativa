import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.config import CASES_PATH, CATEGORIES, Category


class Expected(BaseModel):
    category: Category


Split = Literal["official", "debug"]


class Case(BaseModel):
    id: str
    ticket: str
    expected: Expected
    split: Split = "official"


CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
    },
    "required": ["category"],
    "additionalProperties": False,
}


def load_cases(path: Path = CASES_PATH, split: Split | None = "official") -> list[Case]:
    with open(path, encoding="utf-8") as f:
        cases = [Case(**item) for item in json.load(f)]
    return [c for c in cases if split is None or c.split == split]
