from datetime import datetime
from pathlib import Path

from src.config import RESULTS_PATH
from src.models.base import ModelRunner
from src.pricing import PRICES, cost_usd
from src.prompts import PROMPT_BUILDERS, STRUCTURED_SCHEMAS, extract_answer
from src.results import append_result
from src.schemas import Case
from src.verifier import verify_prediction


def run_case(
    runner: ModelRunner,
    model_key: str,
    case: Case,
    *,
    part: str,
    experiment: str,
    prompt_variant: str = "base",
    run: int = 1,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    effort: str | None = None,
    results_path: Path | None = RESULTS_PATH,
) -> dict:
    price = PRICES.get(model_key, {})
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "part": part,
        "experiment": experiment,
        "model_id": model_key,
        "provider": price.get("provider"),
        "model_name": price.get("model_id"),
        "case_id": case.id,
        "run": run,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "effort": effort,
        "prompt_variant": prompt_variant,
    }

    prompt = PROMPT_BUILDERS[prompt_variant](case.ticket)
    expected = case.expected.model_dump()
    try:
        output = runner.generate(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            effort=effort,
            structured_schema=STRUCTURED_SCHEMAS.get(prompt_variant),
        )
    except NotImplementedError:
        raise
    except Exception as exc:
        record.update(
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            http_status=getattr(exc, "status_code", None),
            raw_output=None,
            parse_ok=False,
            valid_schema=False,
            correct=False,
            predicted=None,
            expected=expected["category"],
        )
    else:
        final_answer = extract_answer(prompt_variant, output["text"])
        record.update(
            status="ok",
            input_tokens=output["input_tokens"],
            output_tokens=output["output_tokens"],
            reasoning_tokens=output["reasoning_tokens"],
            latency_seconds=output["latency_seconds"],
            raw_output=output["text"],
            final_answer=final_answer,
            **({"notices": output["notices"]} if output.get("notices") else {}),
            **verify_prediction(final_answer, expected),
            cost_usd=(
                cost_usd(model_key, output["input_tokens"], output["output_tokens"])
                if model_key in PRICES
                else None
            ),
        )

    if results_path is not None:
        append_result(record, results_path)
    return record
