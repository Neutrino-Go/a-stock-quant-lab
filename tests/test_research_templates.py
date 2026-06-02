from __future__ import annotations

import pytest

from workflow.research_templates import get_research_template


def test_get_research_template_returns_expected_defaults():
    quick = get_research_template("quick")
    standard = get_research_template("standard")
    deep = get_research_template("deep")

    assert quick["top_n_params"] == 3
    assert quick["walk_forward_config"] == {"train_months": 18, "test_months": 6, "step_months": 6}
    assert standard["top_n_params"] == 5
    assert standard["walk_forward_config"]["train_months"] == 24
    assert deep["top_n_params"] == 10
    assert deep["walk_forward_config"] == {"train_months": 36, "test_months": 6, "step_months": 3}


def test_custom_template_is_explicit_and_unknown_template_raises():
    custom = get_research_template("custom")

    assert custom["research_template"] == "custom"
    assert custom["top_n_params"] == 5

    with pytest.raises(ValueError, match="Unknown research template"):
        get_research_template("silent-fallback-is-not-allowed")
