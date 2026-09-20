import json
import re
import subprocess

import pytest

from src.config import RESULTS_PATH, ROOT

if not RESULTS_PATH.exists():
    pytest.skip("results.jsonl no existe: correr los experimentos", allow_module_level=True)

ROWS = [json.loads(line) for line in open(RESULTS_PATH, encoding="utf-8")]
OK = [r for r in ROWS if r["status"] == "ok"]

# modelo, parámetros, corrida, tokens (entrada, salida, razonamiento), latencia, salida y acierto
REQUIRED = ("timestamp", "part", "experiment", "model_id", "provider", "model_name", "case_id", "run", "temperature", "top_p", "top_k",
            "effort", "prompt_variant", "input_tokens", "output_tokens", "reasoning_tokens", "latency_seconds", "raw_output", "correct", "cost_usd")


def test_every_successful_call_has_all_the_columns_the_grader_needs():
    missing = [(i, k) for i, r in enumerate(OK) for k in REQUIRED if k not in r]
    assert not missing, missing[:5]


def test_every_part_of_the_workshop_is_in_the_file():
    assert {"0", "1", "2a", "2b", "3", "4a", "4b"} <= {r["part"] for r in ROWS}


def test_local_generations_are_rows_too_with_cost_zero():
    local = [r for r in OK if r["provider"] in ("ollama", "transformers")]
    assert local and all(r["cost_usd"] == 0.0 for r in local)
    part0 = [r for r in OK if r["part"] == "0"]
    assert part0 and {r["model_id"] for r in part0} == {"gpt2_base"}
    assert {"greedy_reference", "greedy_ignores_temperature", "top_k_1", "degeneration", "base_model_classification"} == {r["experiment"] for r in part0}


def test_failed_calls_keep_the_literal_error_and_status():
    errors = [r for r in ROWS if r["status"] == "error"]
    assert errors and all(r["error_message"] and "http_status" in r for r in errors)


def test_no_credentials_or_provider_ids_in_the_file():
    text = RESULTS_PATH.read_text(encoding="utf-8")
    assert not re.search(r"sk-(proj|ant|live)-[A-Za-z0-9_-]{20,}|org-[A-Za-z0-9]{10,}|proj_[A-Za-z0-9]{8,}|req_[A-Za-z0-9]{10,}|Bearer \S{10,}", text)


def test_git_does_not_ignore_the_file_so_the_grader_can_see_it():
    result = subprocess.run(["git", "check-ignore", "-q", "outputs/raw/results.jsonl"], cwd=ROOT)
    assert result.returncode == 1, "results.jsonl está ignorado por git"
