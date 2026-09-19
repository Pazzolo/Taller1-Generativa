from src.metrics import accuracy
from src.models.base import ModelRunner
from src.models.mock_runner import MockRunner
from src.prompts import build_base_prompt
from src.schemas import Case, load_cases
from src.verifier import verify_prediction


def run(runner: ModelRunner, cases: list[Case]) -> list[dict]:
    records = []
    for case in cases:
        output = runner.generate(build_base_prompt(case.ticket))
        verdict = verify_prediction(output["text"], case.expected.model_dump())
        records.append({"case_id": case.id, "raw_output": output["text"], **verdict})
    return records


def main() -> None:
    cases = load_cases()
    records = run(MockRunner(['{"category": "billing"}']), cases)
    for r in records:
        print(f"{r['case_id']}  expected={r['expected']:<9} predicted={r['predicted']:<9} correct={r['correct']}")
    print(f"accuracy (mock, siempre 'billing'): {accuracy(records):.2f} sobre {len(records)} casos")


if __name__ == "__main__":
    main()
