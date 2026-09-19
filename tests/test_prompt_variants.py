import pytest

from src.config import CATEGORIES
from src.models.openai_runner import OpenAIRunner
from src.prompts import (
    FEW_SHOT_EXAMPLES,
    PROMPT_BUILDERS,
    STRUCTURED_SCHEMAS,
    build_base_prompt,
    cot_prompt,
    extract_answer,
    few_shot_prompt,
    structured_prompt,
    zero_shot_prompt,
)
from src.runner import run_case
from src.schemas import CATEGORY_SCHEMA, load_cases
from tests.fakes import FakeOpenAIClient

KEY = "propietario_economico"
CASE = load_cases()[0]
VARIANTS = ("zero_shot", "few_shot", "cot", "structured")


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_lists_the_categories_and_ends_with_the_ticket(variant):
    prompt = PROMPT_BUILDERS[variant]("my ticket text")
    assert prompt.endswith("my ticket text")
    assert all(c in prompt for c in CATEGORIES)


def test_zero_shot_is_the_base_prompt():
    assert zero_shot_prompt("t") == build_base_prompt("t")


def test_few_shot_examples_never_reuse_an_evaluation_or_debug_case():
    tickets = {c.ticket.strip().lower() for c in load_cases(split=None)}
    for text, _ in FEW_SHOT_EXAMPLES:
        assert text.strip().lower() not in tickets


def test_few_shot_examples_cover_every_category_and_appear_in_the_prompt():
    assert {c for _, c in FEW_SHOT_EXAMPLES} == set(CATEGORIES)
    prompt = few_shot_prompt("t")
    for text, category in FEW_SHOT_EXAMPLES:
        assert text in prompt and f'{{"category": "{category}"}}' in prompt


def test_few_shot_is_longer_than_zero_shot_and_cot_asks_for_steps():
    assert len(few_shot_prompt("t")) > len(zero_shot_prompt("t"))
    assert "step by step" in cot_prompt("t")


def test_structured_prompt_leaves_the_format_to_the_schema():
    assert "JSON" not in structured_prompt("t")
    assert STRUCTURED_SCHEMAS["structured"] == CATEGORY_SCHEMA


def test_extract_answer_takes_the_last_json_object_only_for_cot():
    text = 'The user was charged twice, so probably {"category": "account"}? No, it is money.\n{"category": "billing"}'
    assert extract_answer("cot", text) == '{"category": "billing"}'
    assert extract_answer("zero_shot", text) == text
    assert extract_answer("structured", text) == text


def test_extract_answer_without_json_returns_the_text_unchanged():
    assert extract_answer("cot", "no answer here") == "no answer here"


def test_chatty_zero_shot_answer_fails_the_format_check(tmp_path):
    runner = OpenAIRunner("gpt-4o-mini", client=FakeOpenAIClient(content='Sure! {"category": "billing"}'))
    record = run_case(runner, KEY, CASE, part="3", experiment="t", prompt_variant="zero_shot", results_path=None)
    assert (record["parse_ok"], record["correct"]) == (False, False)


def test_cot_is_verified_on_the_final_answer_and_keeps_the_full_output():
    reasoning = 'Charged twice means a billing problem.\n{"category": "billing"}'
    runner = OpenAIRunner("gpt-4o-mini", client=FakeOpenAIClient(content=reasoning))
    record = run_case(runner, KEY, CASE, part="3", experiment="t", prompt_variant="cot", results_path=None)
    assert record["correct"] is True
    assert record["raw_output"] == reasoning
    assert record["final_answer"] == '{"category": "billing"}'


def test_cot_truncated_before_the_answer_is_recorded_as_a_parse_failure():
    runner = OpenAIRunner("gpt-4o-mini", client=FakeOpenAIClient(content="Let me think about this ticket, first"))
    record = run_case(runner, KEY, CASE, part="3", experiment="t", prompt_variant="cot", results_path=None)
    assert (record["parse_ok"], record["correct"]) == (False, False)


def test_only_the_structured_variant_sends_a_response_format():
    for variant in VARIANTS:
        client = FakeOpenAIClient()
        run_case(OpenAIRunner("gpt-4o-mini", client=client), KEY, CASE, part="3", experiment="t", prompt_variant=variant, results_path=None)
        assert ("response_format" in client.calls[0]) == (variant == "structured")


def test_structured_output_guarantees_format_but_not_the_right_category():
    runner = OpenAIRunner("gpt-4o-mini", client=FakeOpenAIClient(content='{"category": "technical"}'))
    record = run_case(runner, KEY, CASE, part="3", experiment="t", prompt_variant="structured", results_path=None)
    assert record["valid_schema"] is True and record["correct"] is False
