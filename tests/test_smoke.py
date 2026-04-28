"""Import-time smoke tests for talktrace_ai.

Runs as a plain script (`python tests/test_smoke.py`) and under pytest. The
goal is to catch import-level breakage and AppState field drift introduced by
the app.py refactor without exercising any reactive logic.
"""
from __future__ import annotations


def test_app_imports_and_main_callable():
    from talktrace_ai.app import main, app, app_ui, server

    assert callable(main)
    assert app is not None
    assert app_ui is not None
    assert callable(server)


def test_app_state_has_expected_fields():
    from talktrace_ai.state import AppState

    fields = {f.name for f in AppState.__dataclass_fields__.values()}
    expected = {
        "input", "output", "session", "config", "t",
        "transcript_data", "codebook_data", "llm_analysis_data",
        "stats", "current_api", "current_lang", "model",
        "report_a_df", "report_b_df", "run_analysis",
    }
    missing = expected - fields
    assert not missing, f"AppState missing fields: {missing}"


def test_obsidian_css_loads():
    from talktrace_ai.theme import load_obsidian_css

    css = load_obsidian_css()
    assert "data-bs-theme" in css
    assert len(css) > 1000


if __name__ == "__main__":
    test_app_imports_and_main_callable()
    test_app_state_has_expected_fields()
    test_obsidian_css_loads()
    print("smoke tests passed")
