from __future__ import annotations

import pandas as pd
import pytest

from data_quality.adjustment import attach_adjust_type, normalize_adjust_type, validate_adjust_type_consistency


def test_normalize_adjust_type_aliases():
    assert normalize_adjust_type("none") == "raw"
    assert normalize_adjust_type("qfq") == "qfq"
    assert normalize_adjust_type("hfq") == "hfq"


def test_attach_adjust_type():
    frame = attach_adjust_type(pd.DataFrame({"close": [1.0]}), "hfq")

    assert frame["adjust_type"].tolist() == ["hfq"]


def test_mixed_adjust_type_defaults_to_error():
    frame = pd.DataFrame({"adjust_type": ["qfq", "raw"]})

    with pytest.raises(ValueError):
        validate_adjust_type_consistency(frame)


def test_unknown_adjust_type_warning():
    issues = validate_adjust_type_consistency(pd.DataFrame({"adjust_type": ["unknown"]}), raise_on_error=False)

    assert any(issue.issue_type == "adjust_type_mismatch" for issue in issues)


def test_attach_adjust_type_normalizes_alias():
    frame = attach_adjust_type(pd.DataFrame({"close": [1.0]}), "none")

    assert frame["adjust_type"].tolist() == ["raw"]


def test_missing_adjust_type_column_warns_without_raise():
    issues = validate_adjust_type_consistency(pd.DataFrame({"close": [1.0]}), raise_on_error=False)

    assert any(issue.issue_type == "adjust_type_mismatch" for issue in issues)
