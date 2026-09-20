import csv
import importlib.util
import re

import pytest

from src.config import RESULTS_PATH, ROOT, TABLES_DIR

if not RESULTS_PATH.exists():
    pytest.skip("results.jsonl no existe (no se versiona): correr los experimentos", allow_module_level=True)

spec = importlib.util.spec_from_file_location("build_report", ROOT / "scripts" / "build_report.py")
build_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_report)


@pytest.fixture(scope="module")
def built():
    values, tables = build_report.build_values()
    text = build_report.render(build_report.TEMPLATE.read_text(encoding="utf-8"), values, tables)
    return values, tables, text


def test_every_placeholder_in_the_template_resolves(built):
    _, _, text = built
    assert not re.findall(r"\[\[[^\]]+\]\]", text)


def test_headline_numbers_match_the_tables_they_come_from(built):
    values, _, _ = built
    part3 = {r["prompt_variant"]: r for r in csv.DictReader(open(TABLES_DIR / "part3.csv", encoding="utf-8"))}
    assert values["cot_ratio"] == f"{float(part3['cot']['cost_usd_total']) / float(part3['zero_shot']['cost_usd_total']):.1f}"
    part1 = {r["model"]: r for r in csv.DictReader(open(TABLES_DIR / "part1.csv", encoding="utf-8"))}
    ratio = float(part1["openai_razonamiento"]["cost_usd_total"]) / float(part1["propietario_economico"]["cost_usd_total"])
    assert values["p1_luna_ratio"] == f"{ratio:.1f}"


def test_report_has_the_fourteen_sections_in_order(built):
    _, _, text = built
    titles = re.findall(r"^## (\d+)\. ", text, re.M)
    assert titles == [str(i) for i in range(1, 15)]


def test_report_does_not_leak_raw_identifiers_or_ai_tells(built):
    _, _, text = built
    prose = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # las rutas de las figuras sí llevan estos nombres
    for banned in ("high_confidence", "accepted_and_acts", "misleading_frame", "Asimismo", "Cabe destacar", "Por consiguiente", "En síntesis"):
        assert banned not in prose


def test_every_embedded_figure_exists(built):
    _, _, text = built
    paths = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert len(paths) == 6
    assert all((ROOT / "report" / p).resolve().exists() for p in paths)


def test_no_api_key_shaped_string_in_the_report(built):
    _, _, text = built
    assert not re.search(r"sk-(proj|ant|live)-[A-Za-z0-9_-]{20,}", text)


def test_every_answer_with_a_word_limit_from_the_assignment_is_inside_its_range(built):
    _, _, text = built
    blocks = {name: (count, low, high) for name, count, low, high in build_report.word_blocks(text)}
    assert set(blocks) == {"c0c", "c2b", "c3", "c4a", "q1", "q2", "q3", "q4", "q5"}
    assert all(low <= count <= high for count, low, high in blocks.values()), blocks


def test_the_forbidden_word_is_absent_and_the_limits_leave_margin(built):
    _, _, text = built
    assert "creatividad" not in text.lower()
    for name, count, low, high in build_report.word_blocks(text):
        assert low + 5 <= count <= high - 5, (name, count, low, high)  # otro contador de palabras podría discrepar unas pocas


def test_the_report_uses_the_course_table_models_and_not_the_exploratory_one_in_its_tables(built):
    values, tables, _ = built
    for name in ("part1", "part2a", "modelos"):
        assert "gpt-5.5" not in tables[name] and "extra_gpt55" not in tables[name]
    assert "gpt-5.6-luna" in tables["part1"] and "openai_razonamiento" in tables["part2a"]
    assert int(values["n_extra"]) > 0


def test_part_1_table_carries_the_price_verification_date_of_each_row(built):
    _, tables, _ = built
    assert "Precio verificado" in tables["part1"] and "2026-08-26" in tables["part1"] and "2026-09-18" in tables["part1"]
