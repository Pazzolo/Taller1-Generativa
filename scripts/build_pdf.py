"""Convierte report/report.md en report/report.pdf (markdown-it + Chrome headless).

Requiere Google Chrome o Chromium (variable CHROME_BIN si no está en una ruta habitual).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markdown_it import MarkdownIt

from src.config import ROOT

REPORT_DIR = ROOT / "report"
SOURCE = REPORT_DIR / "report.md"
OUTPUT = REPORT_DIR / "report.pdf"

CSS = """
@page { size: A4; margin: 20mm 18mm 22mm 18mm; @bottom-center { content: counter(page); font: 9pt Helvetica, Arial, sans-serif; color: #666; } }
html { font-family: Georgia, "Times New Roman", serif; font-size: 10.5pt; line-height: 1.5; color: #111; }
body { margin: 0; }
h1 { font-size: 21pt; margin: 0 0 .3em; line-height: 1.2; }
h2 { font-size: 14pt; margin: 1.7em 0 .5em; padding-bottom: 3px; border-bottom: 1px solid #c9c8c2; break-after: avoid; }
p { margin: .55em 0; text-align: justify; hyphens: auto; }
ul, ol { margin: .5em 0; padding-left: 1.4em; }
li { margin: .25em 0; }
table { border-collapse: collapse; width: 100%; margin: .8em 0; font: 8pt Helvetica, Arial, sans-serif; break-inside: avoid; }
th, td { border: 1px solid #cfcec8; padding: 2px 5px; text-align: left; vertical-align: top; overflow-wrap: anywhere; }
th { background: #f0efec; }
img { display: block; max-width: 100%; margin: 1em auto .2em; break-inside: avoid; }
p > em:only-child { display: block; font: italic 8.6pt Helvetica, Arial, sans-serif; color: #444; text-align: left; margin-bottom: 1em; }
code { font: 8.6pt Menlo, Consolas, monospace; background: #f4f3f0; padding: 0 2px; border-radius: 2px; overflow-wrap: anywhere; }
pre { background: #f4f3f0; padding: 8px 10px; border-radius: 3px; overflow-x: auto; break-inside: avoid; }
pre code { background: none; padding: 0; }
strong { font-weight: 700; }
"""


def render_html(markdown_text: str, base_dir: Path) -> str:
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render(markdown_text)
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        "<title>Taller 01 — Foundation Models</title>"
        f'<base href="{base_dir.resolve().as_uri()}/"><style>{CSS}</style></head><body>{body}</body></html>'
    )


def find_chrome() -> str:
    candidates = [
        os.environ.get("CHROME_BIN"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("chrome"),
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    raise SystemExit("No se encontró Chrome/Chromium; definir CHROME_BIN con su ruta.")


def wait_for_pdf(process: subprocess.Popen, timeout: float = 120.0) -> None:
    """Chrome en modo headless a veces no termina tras imprimir: se espera a que el archivo deje de crecer y se lo cierra."""
    deadline, last_size, stable = time.time() + timeout, -1, 0
    while time.time() < deadline and process.poll() is None:
        size = OUTPUT.stat().st_size if OUTPUT.exists() else -1
        stable = stable + 1 if size > 10_000 and size == last_size else 0
        if stable >= 3:
            break
        last_size = size
        time.sleep(1)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit("Falta report/report.md: correr scripts/build_report.py primero.")
    html = render_html(SOURCE.read_text(encoding="utf-8"), REPORT_DIR)
    OUTPUT.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        process = subprocess.Popen(
            [find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=10000",
             f"--user-data-dir={tmp}/profile", f"--print-to-pdf={OUTPUT}", page.as_uri()],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        wait_for_pdf(process)
    if not OUTPUT.exists() or OUTPUT.stat().st_size < 10_000:
        raise SystemExit("Chrome no generó el PDF.")
    print(f"{OUTPUT.relative_to(ROOT)} generado ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
