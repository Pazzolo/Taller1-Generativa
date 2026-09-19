import json
from collections import Counter

from src.config import CASES_PATH, CATEGORIES
from src.schemas import Case, load_cases


def test_cases_file_parses_against_schema():
    with open(CASES_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    assert [Case(**item).id for item in raw] == [item["id"] for item in raw]


def test_at_least_ten_official_cases():
    assert len(load_cases()) >= 10


def test_official_set_is_frozen_at_ten():
    assert len(load_cases()) == 10


def test_default_split_excludes_debug_cases():
    assert all(c.split == "official" for c in load_cases())
    assert len(load_cases(split="debug")) == 5
    assert len(load_cases(split=None)) == 15


def test_ids_are_unique():
    ids = [c.id for c in load_cases(split=None)]
    assert len(ids) == len(set(ids))


def test_tickets_are_unique_and_non_empty():
    tickets = [c.ticket.strip().lower() for c in load_cases(split=None)]
    assert all(tickets)
    assert len(tickets) == len(set(tickets))


def test_expected_category_is_in_enum():
    assert all(c.expected.category in CATEGORIES for c in load_cases(split=None))


def test_official_cases_cover_every_category():
    counts = Counter(c.expected.category for c in load_cases())
    assert set(counts) == set(CATEGORIES)
    assert min(counts.values()) >= 3
