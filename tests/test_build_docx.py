import importlib.util
import re
import zipfile

import pytest

pytest.importorskip("docx")
pytest.importorskip("markdown_it")

from src.config import ROOT  # noqa: E402

spec = importlib.util.spec_from_file_location("build_docx", ROOT / "scripts" / "build_docx.py")
build_docx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_docx)

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SAMPLE = """<!--
comentario que no debe verse
-->

# Título del informe

**Estudiante:** Ana Pérez  
**Curso:** Curso de prueba

## 1. Sección

Un párrafo con `código`, **negrita** y *cursiva*.

- primer punto
- segundo punto

1. uno
2. dos

Un párrafo entre las dos listas.

1. otra lista, que debe reiniciar

| A | B |
|---|---|
| 1 | 2 |

![Figura](../outputs/plots/part4_cost_vs_accuracy.png)

*Figura 1. Pie de figura.*

```text
línea 1
línea 2
```
"""


def convert(markdown: str = SAMPLE):
    return build_docx.Converter(ROOT / "report").convert(markdown)


def test_headings_use_word_heading_styles_and_are_bold_at_run_level():
    doc = convert()
    styles = {p.text: p.style.name for p in doc.paragraphs}
    assert styles["Título del informe"] == "Title" and styles["1. Sección"] == "Heading 1"
    section = next(p for p in doc.paragraphs if p.text == "1. Sección")
    assert all(run.bold for run in section.runs)


def test_heading_fonts_do_not_keep_the_theme_attributes_that_word_would_prefer():
    style = convert().styles["Heading 1"].element.rPr.find(W + "rFonts")
    assert style.get(W + "asciiTheme") is None and style.get(W + "ascii") == build_docx.HEADING_FONT


def test_html_comments_never_reach_the_document():
    text = "\n".join(p.text for p in convert().paragraphs)
    assert "comentario" not in text and "<!--" not in text


def test_hard_breaks_keep_the_cover_lines_together():
    cover = next(p for p in convert().paragraphs if p.text.startswith("Estudiante:"))
    assert "Curso: Curso de prueba" in cover.text and "\n" in cover.text


def test_inline_formatting_becomes_runs():
    para = next(p for p in convert().paragraphs if p.text.startswith("Un párrafo"))
    by_text = {r.text: r for r in para.runs}
    assert by_text["negrita"].bold and by_text["cursiva"].italic
    assert by_text["código"].font.name == build_docx.CODE_FONT


def test_lists_use_native_styles_and_each_numbered_list_restarts():
    doc = convert()
    styles = [p.style.name for p in doc.paragraphs]
    assert styles.count("List Bullet") == 2 and styles.count("List Number") == 3
    ids = [p._p.pPr.numPr.numId.val for p in doc.paragraphs if p.style.name == "List Number"]
    assert ids[0] == ids[1] != ids[2]


def test_tables_are_native_with_a_bold_shaded_header():
    doc = convert()
    (table,) = doc.tables
    assert [c.text for c in table.rows[0].cells] == ["A", "B"] and [c.text for c in table.rows[1].cells] == ["1", "2"]
    assert all(run.bold for run in table.rows[0].cells[0].paragraphs[0].runs)
    assert "E9E8E4" in table.rows[0].cells[0]._tc.xml


def test_figures_are_embedded_at_the_page_width_and_captions_follow():
    doc = convert()
    assert len(doc.inline_shapes) == 1 and round(doc.inline_shapes[0].width.cm) == 16
    assert any(p.text.startswith("Figura 1.") for p in doc.paragraphs)


def test_code_blocks_are_monospace_and_shaded():
    para = next(p for p in convert().paragraphs if p.text.startswith("línea 1"))
    assert "línea 2" in para.text and para.runs[0].font.name == build_docx.CODE_FONT
    assert build_docx.CODE_FILL in para._p.xml


def test_document_is_spanish_a4_with_page_numbers(tmp_path):
    doc = convert()
    lang = doc.styles["Normal"].element.rPr.find(W + "lang")
    assert lang.get(W + "val") == "es-ES"
    assert round(doc.sections[0].page_width.cm, 1) == 21.0
    path = tmp_path / "t.docx"
    doc.save(path)
    with zipfile.ZipFile(path) as z:
        assert any("PAGE" in z.read(n).decode() for n in z.namelist() if "footer" in n)


def test_student_name_is_read_from_the_cover():
    assert build_docx.student_name(SAMPLE) == "Ana Pérez"


def test_the_real_report_converts_with_every_table_and_figure():
    source = ROOT / "report" / "report.md"
    if not source.exists():
        pytest.skip("report.md no existe")
    text = source.read_text(encoding="utf-8")
    doc = build_docx.Converter(ROOT / "report").convert(text)
    assert len(doc.tables) == len(re.findall(r"^\|---", text, re.M))
    assert len(doc.inline_shapes) == len(re.findall(r"^!\[", text, re.M)) == 6
    headings = [p.text for p in doc.paragraphs if p.style.name in ("Title", "Heading 1")]
    assert len(headings) == len(re.findall(r"^#{1,2} ", text, re.M))
    assert "son respuestas con límite" not in "\n".join(p.text for p in doc.paragraphs)
