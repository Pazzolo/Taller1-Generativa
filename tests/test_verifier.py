import pytest

from src.verifier import verify_prediction

EXPECTED = {"category": "billing"}


def test_correct_prediction():
    r = verify_prediction('{"category": "billing"}', EXPECTED)
    assert r == {
        "parse_ok": True,
        "valid_schema": True,
        "correct": True,
        "predicted": "billing",
        "expected": "billing",
    }


def test_valid_but_wrong_category():
    r = verify_prediction('{"category": "technical"}', EXPECTED)
    assert (r["parse_ok"], r["valid_schema"], r["correct"]) == (True, True, False)
    assert r["predicted"] == "technical"


def test_category_outside_enum_is_schema_invalid():
    r = verify_prediction('{"category": "refund"}', EXPECTED)
    assert (r["parse_ok"], r["valid_schema"], r["correct"]) == (True, False, False)
    assert r["predicted"] == "refund"


def test_case_sensitive_match():
    r = verify_prediction('{"category": "Billing"}', EXPECTED)
    assert (r["valid_schema"], r["correct"]) == (False, False)


@pytest.mark.parametrize("raw", ["billing", "The category is billing.", "", "{", '{"category": "billing"'])
def test_unparseable_output(raw):
    r = verify_prediction(raw, EXPECTED)
    assert r == {
        "parse_ok": False,
        "valid_schema": False,
        "correct": False,
        "predicted": None,
        "expected": "billing",
    }


@pytest.mark.parametrize("raw", ["[]", '"billing"', "42", "null", '["billing"]'])
def test_valid_json_that_is_not_an_object(raw):
    r = verify_prediction(raw, EXPECTED)
    assert (r["parse_ok"], r["valid_schema"], r["correct"]) == (True, False, False)
    assert r["predicted"] is None


def test_missing_category_key():
    r = verify_prediction('{"label": "billing"}', EXPECTED)
    assert (r["parse_ok"], r["valid_schema"], r["correct"]) == (True, False, False)
    assert r["predicted"] is None


@pytest.mark.parametrize("value", ["null", "5", '["billing"]', "true"])
def test_non_string_category(value):
    r = verify_prediction('{"category": %s}' % value, EXPECTED)
    assert (r["parse_ok"], r["valid_schema"], r["correct"]) == (True, False, False)
    assert r["predicted"] is None


def test_none_raw_output_does_not_raise():
    r = verify_prediction(None, EXPECTED)
    assert r["parse_ok"] is False


def test_extra_keys_are_tolerated_and_category_is_judged():
    r = verify_prediction('{"category": "billing", "reason": "charged twice"}', EXPECTED)
    assert r["correct"] is True
