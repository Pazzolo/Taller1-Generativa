from pathlib import Path
from typing import Literal, get_args

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CASES_PATH = DATA_DIR / "cases.json"
OUTPUTS_DIR = ROOT / "outputs"
RAW_DIR = OUTPUTS_DIR / "raw"
TABLES_DIR = OUTPUTS_DIR / "tables"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_PATH = RAW_DIR / "results.jsonl"

SEED = 42

Category = Literal["billing", "technical", "account"]
CATEGORIES: tuple[str, ...] = get_args(Category)

load_dotenv(ROOT / ".env")
