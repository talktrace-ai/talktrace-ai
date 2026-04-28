"""Smoke tests for codebook_hierarchy + the multi-coding aggregation logic.

The aggregation itself lives inside make_qualitative_stats_df in
talktrace_ai/handlers/results.py, which is tied to the Shiny reactive
system and not easily callable in isolation. So we test:

  1. build_priority_lookup / priority_for cover the resolution rules
  2. A standalone reproduction of the per-turn aggregation, mirroring the
     exact lines in results.py, against synthetic data

Run:
    PYTHONPATH=. python tests/_codebook_hierarchy_smoke.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pandas as pd

from talktrace_ai.utils.codebook_hierarchy import (
    build_priority_lookup,
    priority_for,
    extract_priority_line,
)


# ---------------------------------------------------------------------------
# Hierarchy resolution
# ---------------------------------------------------------------------------

def test_position_based_priority():
    cb = [
        {"Code": "Q1", "Bezeichnung": "Frage1"},
        {"Code": "A1", "Bezeichnung": "Antwort1"},
        {"Code": "B1", "Bezeichnung": "Bewertung1"},
    ]
    lookup = build_priority_lookup(cb)
    assert lookup == {"Q1": 0.0, "A1": 1.0, "B1": 2.0}, lookup
    assert priority_for(lookup, "Q1") == 0.0
    assert priority_for(lookup, "A1") == 1.0
    assert priority_for(lookup, "Z9") >= 1_000_000  # unknown sorts last
    print("OK position-based priority")


def test_explicit_priority_column():
    cb = [
        {"Code": "Q1", "Priorität": "5"},
        {"Code": "A1", "Priorität": "1"},
        {"Code": "B1", "Priorität": 3},
    ]
    lookup = build_priority_lookup(cb)
    assert lookup == {"Q1": 5.0, "A1": 1.0, "B1": 3.0}, lookup
    print("OK explicit Priorität column wins over position")


def test_mixed_explicit_and_position():
    """Some entries have explicit priority, others fall back to position."""
    cb = [
        {"Code": "Q1"},                     # position 0 → priority 0
        {"Code": "A1", "Priorität": "5"},   # explicit 5
        {"Code": "B1"},                     # position 2 → priority 2
    ]
    lookup = build_priority_lookup(cb)
    assert lookup == {"Q1": 0.0, "A1": 5.0, "B1": 2.0}, lookup
    print("OK mixed explicit/positional priority")


def test_english_priority_key():
    cb = [
        {"Code": "Q1", "Priority": "2.5"},
        {"Code": "A1", "Priority": "0.5"},
    ]
    lookup = build_priority_lookup(cb)
    assert lookup == {"Q1": 2.5, "A1": 0.5}, lookup
    print("OK English 'Priority' column recognised")


def test_invalid_priority_falls_back_to_position():
    cb = [
        {"Code": "Q1", "Priorität": "nicht-zahl"},
        {"Code": "A1", "Priorität": ""},
        {"Code": "B1"},
    ]
    lookup = build_priority_lookup(cb)
    # All invalid/empty/missing → position
    assert lookup == {"Q1": 0.0, "A1": 1.0, "B1": 2.0}, lookup
    print("OK invalid priority values fall back to position")


def test_alternative_code_keys():
    """Codebooks may use 'Shortcode' instead of 'Code' as the identifier."""
    cb = [
        {"Shortcode": "X", "Bezeichnung": "Foo"},
        {"shortcode": "Y", "label": "bar"},
    ]
    lookup = build_priority_lookup(cb)
    assert "X" in lookup and "Y" in lookup, lookup
    assert lookup["X"] == 0.0 and lookup["Y"] == 1.0, lookup
    print("OK alternative code-key spellings")


def test_empty_or_invalid_codebook():
    assert build_priority_lookup(None) == {}
    assert build_priority_lookup([]) == {}
    assert build_priority_lookup("not a list") == {}
    print("OK degenerate codebook inputs return empty lookup")


# ---------------------------------------------------------------------------
# Priority line ("Priorisierung: A1 > B2 > C3")
# ---------------------------------------------------------------------------

def test_priority_line_with_label_gt():
    cb = [
        {"Code": "A1", "Bezeichnung": "Antwort"},
        {"Code": "B2", "Bezeichnung": "Bewertung"},
        {"Code": "C3", "Bezeichnung": "Frage", "Beschreibung": "Priorisierung: C3 > A1 > B2"},
    ]
    line = extract_priority_line(cb)
    assert line == ["C3", "A1", "B2"], line
    lookup = build_priority_lookup(cb)
    assert lookup["C3"] == 0.0
    assert lookup["A1"] == 1.0
    assert lookup["B2"] == 2.0
    print("OK priority line 'Priorisierung: C3 > A1 > B2' parsed and overrides position")


def test_priority_line_arrow_separator():
    cb = [
        {"Code": "X1"},
        {"Code": "X2"},
        {"Code": "Note", "Bezeichnung": "Priority: X2 → X1"},
    ]
    line = extract_priority_line(cb)
    assert line == ["X2", "X1"], line
    print("OK priority line accepts arrow separator")


def test_priority_line_comma_separator():
    cb = [
        {"Code": "Q"},
        {"Code": "A"},
        {"Code": "B"},
        {"Code": "Note", "Bezeichnung": "Hierarchie: B, Q, A"},
    ]
    line = extract_priority_line(cb)
    assert line == ["B", "Q", "A"], line
    print("OK priority line accepts ',' separator")


def test_priority_line_without_label():
    """Eine reine Sequenz ohne Label-Wort wird auch akzeptiert, wenn sie
    aus mind. zwei bekannten Codes besteht."""
    cb = [
        {"Code": "Q1"},
        {"Code": "A1"},
        {"Code": "B1"},
        {"Code": "ord", "Bezeichnung": "B1 > Q1 > A1"},
    ]
    line = extract_priority_line(cb)
    assert line == ["B1", "Q1", "A1"], line
    print("OK priority line accepted without explicit label keyword")


def test_priority_line_ignores_unknown_tokens():
    """Tokens, die nicht im Codebuch sind, werden still ignoriert."""
    cb = [
        {"Code": "A"},
        {"Code": "B"},
        {"Code": "ord", "Bezeichnung": "Priorisierung: A > UNKNOWN > B > Z9"},
    ]
    line = extract_priority_line(cb)
    assert line == ["A", "B"], line
    print("OK unknown tokens in priority line are skipped silently")


def test_priority_line_needs_two_known_codes():
    """Eine Zeile mit nur einem bekannten Code wird NICHT als Priorisierung erkannt."""
    cb = [
        {"Code": "A"},
        {"Code": "ord", "Bezeichnung": "Priorisierung: A > UNKNOWN_X"},
    ]
    line = extract_priority_line(cb)
    assert line is None, line
    print("OK priority line ignored when fewer than 2 codes match")


def test_priority_line_with_priority_column_mix():
    """Codes in der Zeile gewinnen vor explizit gesetzter Spalte;
    Codes ausserhalb der Zeile nutzen weiter ihre Spalten-Priorität."""
    cb = [
        {"Code": "Q1", "Priorität": "10"},   # nicht in Zeile → 0 + 10 = 10
        {"Code": "A1", "Priorität": "1"},    # in Zeile, Position 0 → 0
        {"Code": "B1", "Priorität": "2"},    # in Zeile, Position 1 → 1
        {"Code": "ord", "Bezeichnung": "Priorisierung: A1 > B1"},
    ]
    lookup = build_priority_lookup(cb)
    assert lookup["A1"] == 0.0
    assert lookup["B1"] == 1.0
    # Q1 ist nicht in der Zeile → seine explizite 10 wird um Offset (=2) verschoben
    assert lookup["Q1"] == 12.0
    print("OK priority line + explicit column: line wins, others are offset")


def test_priority_line_beats_position():
    """Wenn die Zeile A1 > B2 > C3 sagt, aber im Codebook B2 zuerst steht,
    gewinnt trotzdem die Reihenfolge der Zeile."""
    cb = [
        {"Code": "B2", "Bezeichnung": "B-Code"},
        {"Code": "C3", "Bezeichnung": "C-Code"},
        {"Code": "A1", "Bezeichnung": "A-Code"},
        {"Code": "_priority_", "Bezeichnung": "Priorisierung: A1 > B2 > C3"},
    ]
    lookup = build_priority_lookup(cb)
    # Zeile: A1=0, B2=1, C3=2 — egal wo sie im Codebook stehen.
    assert lookup["A1"] == 0.0
    assert lookup["B2"] == 1.0
    assert lookup["C3"] == 2.0
    print("OK priority line overrides codebook position order")


# ---------------------------------------------------------------------------
# Aggregation reproduction (the core multi-coding logic from results.py)
# ---------------------------------------------------------------------------

def _aggregate_like_results(coded, codebook, multi_coding):
    """Mirrors the new lines in results.py:make_qualitative_stats_df.

    Takes a 'coded' DataFrame with at least Shortcode and __key__ columns,
    and reproduces the priority-aware deduplication / multi-coding join.
    """
    coded = coded.copy()
    lookup = build_priority_lookup(codebook)
    coded["__priority__"] = coded["Shortcode"].apply(
        lambda c: priority_for(lookup, str(c).strip())
    )
    coded = coded.sort_values("__priority__", kind="mergesort")
    if multi_coding:
        coded = (
            coded.groupby("__key__", sort=False)
                 .agg({"Shortcode": lambda s: "; ".join(
                     dict.fromkeys(str(c).strip() for c in s if str(c).strip())
                 )})
                 .reset_index()
        )
    else:
        coded = coded.drop_duplicates(subset=["__key__"], keep="first")
        coded = coded.drop(columns=["__priority__"])
    return coded


def test_single_coding_keeps_highest_priority():
    """Same turn coded twice; highest-priority code wins."""
    cb = [
        {"Code": "Q1"},   # priority 0 (highest)
        {"Code": "A1"},   # priority 1
        {"Code": "B1"},   # priority 2
    ]
    coded = pd.DataFrame([
        {"__key__": "L :: hi", "Shortcode": "B1"},  # emitted first by LLM
        {"__key__": "L :: hi", "Shortcode": "Q1"},  # emitted second, but wins on priority
        {"__key__": "L :: yo", "Shortcode": "A1"},
    ])
    out = _aggregate_like_results(coded, cb, multi_coding=False)
    out_dict = dict(zip(out["__key__"], out["Shortcode"]))
    assert out_dict == {"L :: hi": "Q1", "L :: yo": "A1"}, out_dict
    print("OK single-coding picks highest-priority code per turn")


def test_multi_coding_joins_in_priority_order():
    cb = [
        {"Code": "Q1"},   # 0
        {"Code": "A1"},   # 1
        {"Code": "B1"},   # 2
    ]
    coded = pd.DataFrame([
        {"__key__": "L :: hi", "Shortcode": "B1"},  # priority 2
        {"__key__": "L :: hi", "Shortcode": "A1"},  # priority 1
        {"__key__": "L :: hi", "Shortcode": "Q1"},  # priority 0
    ])
    out = _aggregate_like_results(coded, cb, multi_coding=True)
    assert len(out) == 1, out
    sc = out.iloc[0]["Shortcode"]
    assert sc == "Q1; A1; B1", f"expected Q1; A1; B1 in priority order, got {sc!r}"
    print("OK multi-coding joins codes in priority order")


def test_multi_coding_dedupes_repeated_codes():
    """If the LLM emits the same code twice for one turn, the result has it only once."""
    cb = [{"Code": "Q1"}, {"Code": "A1"}]
    coded = pd.DataFrame([
        {"__key__": "L :: hi", "Shortcode": "Q1"},
        {"__key__": "L :: hi", "Shortcode": "Q1"},
        {"__key__": "L :: hi", "Shortcode": "A1"},
    ])
    out = _aggregate_like_results(coded, cb, multi_coding=True)
    sc = out.iloc[0]["Shortcode"]
    assert sc == "Q1; A1", f"expected deduped 'Q1; A1', got {sc!r}"
    print("OK multi-coding dedupes repeated codes per turn")


def test_explicit_priority_overrides_position_in_aggregation():
    """A code with high explicit priority beats a code that comes earlier in the list."""
    cb = [
        {"Code": "Q1"},                       # position 0
        {"Code": "A1", "Priorität": "-5"},   # explicit -5 → highest priority
    ]
    coded = pd.DataFrame([
        {"__key__": "L :: hi", "Shortcode": "Q1"},
        {"__key__": "L :: hi", "Shortcode": "A1"},
    ])
    out_single = _aggregate_like_results(coded, cb, multi_coding=False)
    assert out_single.iloc[0]["Shortcode"] == "A1", out_single
    out_multi = _aggregate_like_results(coded, cb, multi_coding=True)
    assert out_multi.iloc[0]["Shortcode"] == "A1; Q1", out_multi
    print("OK explicit priority beats position in both modes")


def test_unknown_code_sorts_last():
    """A halluzinierter code (not in codebook) goes to the end of the priority list."""
    cb = [{"Code": "Q1"}, {"Code": "A1"}]
    coded = pd.DataFrame([
        {"__key__": "L :: hi", "Shortcode": "ZZ"},  # unknown
        {"__key__": "L :: hi", "Shortcode": "A1"},
    ])
    out_single = _aggregate_like_results(coded, cb, multi_coding=False)
    assert out_single.iloc[0]["Shortcode"] == "A1", out_single  # known wins over unknown
    out_multi = _aggregate_like_results(coded, cb, multi_coding=True)
    assert out_multi.iloc[0]["Shortcode"] == "A1; ZZ", out_multi
    print("OK unknown codes sort after known codes")


def test_bar_plot_split_explode_round_trip():
    """Simulate the bar-plot split/explode logic on a multi-coded merged df."""
    merged = pd.DataFrame({
        "Shortcode": ["RE; A; CO", "RE", "A; CO", ""]
    })
    series = merged["Shortcode"].astype(str).str.strip()
    series = series.str.split(r"\s*;\s*", regex=True)
    exploded = series.explode().str.strip()
    exploded = exploded[exploded != ""]
    counts = exploded.value_counts().to_dict()
    assert counts == {"RE": 2, "A": 2, "CO": 2}, counts
    print("OK bar-plot split+explode counts each code separately")


if __name__ == "__main__":
    test_position_based_priority()
    test_explicit_priority_column()
    test_mixed_explicit_and_position()
    test_english_priority_key()
    test_invalid_priority_falls_back_to_position()
    test_alternative_code_keys()
    test_empty_or_invalid_codebook()
    test_priority_line_with_label_gt()
    test_priority_line_arrow_separator()
    test_priority_line_comma_separator()
    test_priority_line_without_label()
    test_priority_line_ignores_unknown_tokens()
    test_priority_line_needs_two_known_codes()
    test_priority_line_with_priority_column_mix()
    test_priority_line_beats_position()
    test_single_coding_keeps_highest_priority()
    test_multi_coding_joins_in_priority_order()
    test_multi_coding_dedupes_repeated_codes()
    test_explicit_priority_overrides_position_in_aggregation()
    test_unknown_code_sorts_last()
    test_bar_plot_split_explode_round_trip()
    print("\nall codebook hierarchy + multi-coding tests passed")
