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


def test_theme_css_loads():
    from talktrace_ai.theme import load_theme_css

    css = load_theme_css()
    assert "data-bs-theme" in css
    assert len(css) > 1000


def test_client_factories_build():
    """Catches NameErrors hidden behind force_no_llm=True demo paths.

    Each get_*_client function uses the module-level _client_cache; if a
    refactor moves the cache without re-binding here, the smoke test fails
    instead of crashing in production on the first real API call.
    """
    from talktrace_ai.utils.llm_clients import (
        get_groq_client, get_openai_client, get_anthropic_client,
    )

    assert type(get_groq_client("sk-test")).__name__ == "Groq"
    assert type(get_openai_client("sk-test")).__name__ == "OpenAI"
    assert type(get_anthropic_client("sk-test")).__name__ == "Anthropic"


def test_cache_key_resolves_format_codebook():
    """Catches the cross-package NameError for _format_codebook.

    _cache_key lives in utils.llm_cache and lazy-imports _format_codebook
    from utils.llm_analysis._json to avoid a load-time cycle. If that
    import path breaks, every real LLM call would crash at the first
    cache-key computation; this test surfaces that immediately.
    """
    from talktrace_ai.utils.llm_cache import _cache_key

    h = _cache_key("openai", "gpt-4", "sys", "usr", "transcript", None)
    assert isinstance(h, str)
    assert len(h) == 32  # md5 hexdigest


def test_llm_analysis_provider_subpackage():
    """All four providers must remain importable from the public path."""
    from talktrace_ai.utils.llm_analysis import (
        llm_analysis_groq, llm_analysis_openai,
        llm_analysis_anthropic, llm_analysis_ollama,
    )

    for fn in (llm_analysis_groq, llm_analysis_openai,
               llm_analysis_anthropic, llm_analysis_ollama):
        assert callable(fn)


def test_handler_sections_export_register():
    """Each of the six handler section modules exports a callable register."""
    from talktrace_ai.handlers import (
        onboarding, sidebar, analysis, testing, results, options,
    )
    for mod in (onboarding, sidebar, analysis, testing, results, options):
        assert callable(getattr(mod, "register", None)), (
            f"{mod.__name__} missing callable register(state)"
        )


def _impulse_df(codes_by_impulse):
    import pandas as pd
    rows = [{"Sprecher": "S1", "Impuls": imp, "Shortcode": code}
            for imp, code in codes_by_impulse.items()]
    return pd.DataFrame(rows, columns=["Sprecher", "Impuls", "Shortcode"])


def test_intercoder_multi_perfect_agreement():
    """3 raters in total agreement → Fleiss / Krippendorff ≈ 1.0."""
    from talktrace_ai.utils.intercoder import compute_intercoder_agreement_multi

    coding = {f"i{i}": ("A" if i % 2 else "B") for i in range(20)}
    dfs = [_impulse_df(coding) for _ in range(3)]

    res_f = compute_intercoder_agreement_multi(dfs, metric="fleiss")
    assert abs(res_f["value"] - 1.0) < 1e-9
    assert res_f["n_raters"] == 3
    assert res_f["n_units"] == 20

    res_k = compute_intercoder_agreement_multi(dfs, metric="krippendorff")
    assert abs(res_k["value"] - 1.0) < 1e-9


def test_intercoder_multi_random_low_agreement():
    """Independent random codings should give a metric near 0 (not significant)."""
    import random
    from talktrace_ai.utils.intercoder import compute_intercoder_agreement_multi

    random.seed(7)
    impulses = [f"i{i}" for i in range(50)]
    codes = ["A", "B", "C"]
    dfs = []
    for _ in range(3):
        dfs.append(_impulse_df({imp: random.choice(codes) for imp in impulses}))

    res = compute_intercoder_agreement_multi(dfs, metric="fleiss",
                                             n_boot=200, seed=42)
    # Random raters: Fleiss kappa should be small in absolute value.
    assert abs(res["value"]) < 0.3


def test_intercoder_multi_cohen_matches_basic():
    """Cohen's κ via expert path equals the basic 2-rater compute (same data)."""
    from talktrace_ai.utils.intercoder import (
        compute_intercoder_agreement, compute_intercoder_agreement_multi,
    )
    coding_a = {f"i{i}": ("A" if i < 7 else "B") for i in range(10)}
    coding_b = {f"i{i}": ("A" if i < 5 else "B") for i in range(10)}
    df_a = _impulse_df(coding_a)
    df_b = _impulse_df(coding_b)

    basic = compute_intercoder_agreement(df_a, df_b)
    expert = compute_intercoder_agreement_multi([df_a, df_b], metric="cohen")
    assert abs(basic["kappa"] - expert["value"]) < 1e-9


def test_intercoder_multi_metric_validation():
    """Cohen requires N=2; Fleiss requires N≥3."""
    from talktrace_ai.utils.intercoder import compute_intercoder_agreement_multi
    df = _impulse_df({"i1": "A", "i2": "B"})
    try:
        compute_intercoder_agreement_multi([df, df, df], metric="cohen")
        raised = False
    except ValueError:
        raised = True
    assert raised, "cohen with N=3 must raise ValueError"
    try:
        compute_intercoder_agreement_multi([df, df], metric="fleiss")
        raised = False
    except ValueError:
        raised = True
    assert raised, "fleiss with N=2 must raise ValueError"


def test_p_value_stars():
    from talktrace_ai.utils.intercoder import p_value_stars
    assert p_value_stars(0.0001) == "***"
    assert p_value_stars(0.005) == "**"
    assert p_value_stars(0.03) == "*"
    assert p_value_stars(0.2) == "n.s."
    assert p_value_stars(float("nan")) == "n.s."


if __name__ == "__main__":
    test_app_imports_and_main_callable()
    test_app_state_has_expected_fields()
    test_theme_css_loads()
    test_client_factories_build()
    test_cache_key_resolves_format_codebook()
    test_llm_analysis_provider_subpackage()
    test_handler_sections_export_register()
    test_intercoder_multi_perfect_agreement()
    test_intercoder_multi_random_low_agreement()
    test_intercoder_multi_cohen_matches_basic()
    test_intercoder_multi_metric_validation()
    test_p_value_stars()
    print("smoke tests passed")
