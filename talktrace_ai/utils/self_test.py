"""Gold-standard self-test — deterministic checks against a known fixture.

The point is to give the user a quick way to verify the app's *calculations*
(participation counts, turn stats, intercoder metrics, fingerprint) before
they trust it on their own data. We deliberately do not call any LLM here:
LLM output isn't deterministic, so a self-test that depends on it can't tell
the user whether *the app* is broken or whether *the model* hallucinated.
The fixture comes from talktrace_ai.examples.demo, which already ships an
expected coding for the synthetic transcript.

Each check returns ``(label, status, expected, actual, detail)`` where
``status in {"pass", "fail", "skip"}``.
"""
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from ..examples.demo import (
    DEMO_TRANSCRIPT, DEMO_TEACHER_NAME, build_demo_llm_analysis_df,
)
from .stats import count_pupils, dialog_stats
from .intercoder import compute_intercoder_agreement
from .fingerprint import compute_fingerprint


def _approx_eq(a, b, tol=1e-6) -> bool:
    if a is None or b is None:
        return False
    try:
        return math.isclose(float(a), float(b), abs_tol=tol)
    except (TypeError, ValueError):
        return False


def _result(label, ok, expected, actual, detail=""):
    return {
        "label": label,
        "status": "pass" if ok else "fail",
        "expected": expected,
        "actual": actual,
        "detail": detail,
    }


def run_self_test(lang: str = "de") -> dict:
    """Run all self-test checks. Returns dict with overall + per-check rows."""
    transcript = DEMO_TRANSCRIPT[lang]
    teacher = DEMO_TEACHER_NAME[lang]
    expected_df = build_demo_llm_analysis_df(lang)

    checks = []

    # -- count_pupils: demo transcript has S01..S04 -----------------------
    try:
        n_pupils = count_pupils(transcript)
        checks.append(_result(
            "count_pupils on demo transcript", n_pupils == 4, 4, n_pupils,
        ))
    except Exception as e:
        checks.append(_result("count_pupils on demo transcript", False, 4,
                              None, detail=str(e)))

    # -- dialog_stats: teacher and student turn counts --------------------
    try:
        df_stats = dialog_stats(transcript, teacher)
        teacher_row = df_stats.loc[df_stats["Sprecher"] == teacher]
        student_row = df_stats.loc[df_stats["Sprecher"] == "Schüler:innen"]
        teacher_n = int(teacher_row["Anzahl_Beitraege"].iloc[0]) if not teacher_row.empty else None
        student_n = int(student_row["Anzahl_Beitraege"].iloc[0]) if not student_row.empty else None
        # Ground truth (counted from talktrace_ai.examples.demo): 13 teacher
        # turns, 9 student turns.
        checks.append(_result(
            "dialog_stats teacher turn count",
            teacher_n == 13, 13, teacher_n,
        ))
        checks.append(_result(
            "dialog_stats student turn count",
            student_n == 9, 9, student_n,
        ))
    except Exception as e:
        checks.append(_result("dialog_stats", False, "(13 teacher, 9 student)",
                              None, detail=str(e)))

    # -- intercoder: identical reports → κ=1.0, percent_agreement=1.0 -----
    try:
        identical = compute_intercoder_agreement(expected_df.copy(),
                                                 expected_df.copy())
        ok = (_approx_eq(identical.get("kappa"), 1.0)
              and _approx_eq(identical.get("percent_agreement"), 1.0)
              and _approx_eq(identical.get("krippendorff_alpha"), 1.0)
              and _approx_eq(identical.get("gwet_ac1"), 1.0)
              and _approx_eq(identical.get("brennan_prediger"), 1.0))
        checks.append(_result(
            "intercoder agreement on identical reports",
            ok, "κ = α = AC1 = BP = 1.000",
            (f"κ={identical.get('kappa'):.3f}, "
             f"α={identical.get('krippendorff_alpha'):.3f}, "
             f"AC1={identical.get('gwet_ac1'):.3f}, "
             f"BP={identical.get('brennan_prediger'):.3f}"),
        ))
    except Exception as e:
        checks.append(_result("intercoder agreement on identical reports",
                              False, "κ = 1.000", None, detail=str(e)))

    # -- intercoder: perturbed report → κ < 1.0 strictly between 0 and 1 --
    try:
        perturbed = expected_df.copy()
        # Flip the first three Shortcodes deterministically so we know
        # disagreement is non-trivial but not catastrophic.
        if len(perturbed) >= 3:
            for i in range(3):
                cur = perturbed.at[i, "Shortcode"]
                perturbed.at[i, "Shortcode"] = "F1" if cur != "F1" else "Q1"
        perturbed_res = compute_intercoder_agreement(expected_df.copy(),
                                                    perturbed)
        k = perturbed_res.get("kappa")
        # We just need monotonicity here: 3 flips out of 22 → κ should be
        # strictly < 1, and well above 0 (most codes still agree).
        ok = k is not None and 0.5 <= k < 1.0
        checks.append(_result(
            "intercoder agreement drops on perturbation",
            ok, "0.5 ≤ κ < 1.0",
            f"κ = {k:.3f}" if k is not None else "n/a",
        ))
    except Exception as e:
        checks.append(_result("intercoder agreement drops on perturbation",
                              False, "0.5 ≤ κ < 1.0", None, detail=str(e)))

    # -- fingerprint: deterministic, 12 hex chars -------------------------
    try:
        fp1 = compute_fingerprint("cb", "sys", "usr", "model", transcript)
        fp2 = compute_fingerprint("cb", "sys", "usr", "model", transcript)
        fp3 = compute_fingerprint("cb", "sys", "usr", "model-other", transcript)
        ok = (len(fp1) == 12 and fp1 == fp2 and fp1 != fp3
              and all(c in "0123456789abcdef" for c in fp1))
        checks.append(_result(
            "reproducibility fingerprint is deterministic",
            ok, "12 hex chars; equal for equal inputs; differs on model change",
            f"fp1={fp1}, fp2={fp2}, fp3={fp3}",
        ))
    except Exception as e:
        checks.append(_result("reproducibility fingerprint is deterministic",
                              False, "12 hex chars", None, detail=str(e)))

    n_pass = sum(1 for c in checks if c["status"] == "pass")
    n_total = len(checks)
    return {
        "checks": checks,
        "n_pass": n_pass,
        "n_total": n_total,
        "all_pass": n_pass == n_total,
    }
