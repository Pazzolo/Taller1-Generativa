import importlib.util
import subprocess
import sys

import pytest

from src.config import COURSE_MODELS, ROOT

spec = importlib.util.spec_from_file_location("run_all", ROOT / "scripts" / "run_all.py")
run_all = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_all)


@pytest.fixture
def recorded(monkeypatch):
    """Sustituye main() de cada experimento por un registro de los argumentos que recibiría."""
    calls = []
    for part in run_all.PARTS:
        module = importlib.import_module(f"src.experiments.part{part}")
        monkeypatch.setattr(module, "main", lambda part=part: calls.append((part, list(sys.argv[1:]))))
    return calls


def test_part_1_runs_each_of_the_three_course_models(recorded):
    run_all.run_part("1")
    assert recorded == [("1", ["--model", m]) for m in COURSE_MODELS]


def test_part_4a_runs_the_sweep_and_the_control_and_4b_the_intermediate_levels_too(recorded):
    run_all.run_part("4a")
    run_all.run_part("4b")
    assert recorded == [("4a", []), ("4a", ["--control"]), ("4b", []), ("4b", ["--levels", "low", "high"])]


def test_dry_run_is_forwarded_only_to_parts_that_support_it(recorded):
    for part in run_all.PARTS:
        run_all.run_part(part, dry_run=True)
    forwarded = {part for part, argv in recorded if "--dry-run" in argv}
    assert forwarded == set(run_all.SUPPORTS_DRY_RUN)
    assert not any(part in ("0", "1") for part, _ in recorded)  # esas se omiten en dry-run, no se ejecutan


def test_every_part_has_a_plan_and_all_are_covered():
    assert set(run_all.PLAN) == set(run_all.PARTS)


@pytest.mark.parametrize("part", ["2a", "2b", "3", "4a", "4b"])
def test_documented_command_really_works_end_to_end_without_calling_any_api(part):
    """Este es el comando del README; antes fallaba con «unrecognized arguments»."""
    result = subprocess.run([sys.executable, "scripts/run_all.py", "--part", part, "--dry-run"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-400:]
    assert "calls planned" in result.stdout
