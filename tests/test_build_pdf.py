import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("markdown_it")

from src.config import ROOT  # noqa: E402

spec = importlib.util.spec_from_file_location("build_pdf", ROOT / "scripts" / "build_pdf.py")
build_pdf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_pdf)

SAMPLE = """# Título

Un párrafo con `código` y **negrita**.

| A | B |
|---|---|
| 1 | 2 |

![Figura](../outputs/plots/part4_cost_vs_accuracy.png)

*Figura 1. Pie de figura.*
"""


def test_html_has_language_encoding_and_a_base_for_relative_images():
    html = build_pdf.render_html(SAMPLE, Path("/tmp/report"))
    assert '<html lang="es">' in html and '<meta charset="utf-8">' in html
    assert '<base href="file:///' in html and "/report/" in html


def test_markdown_tables_and_images_become_html():
    html = build_pdf.render_html(SAMPLE, Path("/tmp/report"))
    assert "<table>" in html and "<th>A</th>" in html and "<td>2</td>" in html
    assert '<img src="../outputs/plots/part4_cost_vs_accuracy.png"' in html
    assert "<em>Figura 1. Pie de figura.</em>" in html


def test_page_numbers_and_a4_are_requested_in_the_stylesheet():
    css = build_pdf.CSS
    assert "size: A4" in css and "counter(page)" in css


def test_real_report_renders_all_tables_and_figures_if_it_exists():
    source = ROOT / "report" / "report.md"
    if not source.exists():
        pytest.skip("report.md no existe")
    text = source.read_text(encoding="utf-8")
    html = build_pdf.render_html(text, ROOT / "report")
    assert html.count("<img ") == 6
    assert html.count("<table>") == sum(1 for line in text.splitlines() if line.startswith("|---") or line.startswith("|--"))
    assert "[[" not in html
