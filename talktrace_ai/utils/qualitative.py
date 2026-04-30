"""Pure helpers for the quantitative + qualitative plots / tables.

Both the manual results pipeline (handlers/results.py) and the autopilot
report builder (handlers/autopilot.py) need to compute the same per-coding
artefacts: the merged 'qualitative stats' DataFrame and its bar plot, plus
the speaker-words distribution plot. Extracting these into pure functions
lets the autopilot generate per-coder reports without temporarily mutating
the global reactive state.
"""
import re

import matplotlib.pyplot as plt
import pandas as pd

from .codebook_hierarchy import build_priority_lookup, priority_for
from .plot_style import (
    apply_axes_style,
    primary_color,
    round_bar_corners,
    style_no_data_axes,
)
from .stats import _parse_turns


def build_qual_stats_df(
    analysis_df,
    transcript_text,
    teacher_name,
    codebook_data,
    multi_coding,
    t,
    converted_transcript=None,
):
    """Merge an LLM coding DataFrame with the parsed transcript turns.

    Mirrors handlers/results.py:make_qualitative_stats_df but takes all
    inputs as plain values so callers don't need to be inside a reactive
    context. Returns the merged DataFrame with the localized column names.
    """
    cols = [
        "#",
        t("report", "speaker"),
        t("report", "teacher_statement"),
        t("report", "shortcode"),
    ]
    if analysis_df is None:
        return pd.DataFrame(columns=cols)
    analysis_df = analysis_df.copy()
    if analysis_df.empty:
        return pd.DataFrame(columns=cols)
    if "Sprecher" not in analysis_df.columns:
        analysis_df["Sprecher"] = ""

    if not transcript_text and converted_transcript:
        if isinstance(converted_transcript, dict):
            transcript_text = converted_transcript.get("text")

    if not transcript_text:
        analysis_df["#"] = analysis_df.reset_index().index + 1
        analysis_df = analysis_df[["#", "Sprecher", "Impuls", "Shortcode"]]
        analysis_df.columns = cols
        return analysis_df

    turns = _parse_turns(transcript_text, teacher_name)
    all_turns_df = pd.DataFrame(turns, columns=["Sprecher", "Impuls"])
    all_turns_df["Sprecher"] = all_turns_df["Sprecher"].apply(
        lambda s: teacher_name if s.lower() == teacher_name.lower() else s
    )
    all_turns_df["#"] = range(1, len(all_turns_df) + 1)

    def _norm_impuls(s):
        t_ = re.sub(r"\s+", " ", str(s)).strip()
        return re.sub(
            r"^[\s\"'„“”»«()\[\]\.…!?,:;-]+|[\s\"'„“”»«()\[\]\.…!?,:;-]+$",
            "",
            t_,
        ).lower()

    all_turns_df["__key__"] = (
        all_turns_df["Sprecher"] + " :: " + all_turns_df["Impuls"].apply(_norm_impuls)
    )
    coded = analysis_df[["Sprecher", "Impuls", "Shortcode"]].copy()
    _teacher_aliases = {"lehrperson", "lehrer", "lehrkraft", "teacher",
                        teacher_name.lower()}
    coded["Sprecher"] = coded["Sprecher"].apply(
        lambda s: teacher_name if str(s).lower() in _teacher_aliases else s
    )
    coded["__key__"] = (
        coded["Sprecher"] + " :: " + coded["Impuls"].apply(_norm_impuls)
    )
    _priority_lookup = build_priority_lookup(codebook_data)
    coded["__priority__"] = coded["Shortcode"].apply(
        lambda c: priority_for(_priority_lookup, str(c).strip())
    )
    coded = coded.sort_values("__priority__", kind="mergesort")
    if multi_coding:
        coded = (
            coded.groupby("__key__", sort=False)
            .agg({
                "Shortcode": lambda s: "; ".join(
                    dict.fromkeys(str(c).strip() for c in s if str(c).strip())
                )
            })
            .reset_index()
        )
    else:
        coded = coded.drop_duplicates(subset=["__key__"], keep="first")
        coded = coded.drop(columns=["__priority__"])
    merged = pd.merge(
        all_turns_df,
        coded[["__key__", "Shortcode"]],
        on="__key__",
        how="left",
    )
    merged = merged.drop(columns=["__key__"])
    merged = merged[["#", "Sprecher", "Impuls", "Shortcode"]].copy()
    merged["Shortcode"] = merged["Shortcode"].fillna("").astype(str)
    merged.columns = cols
    return merged


def build_qual_plot(merged_df, t, mode="light"):
    """Bar plot of code frequencies. Returns a matplotlib axes (always)."""
    if merged_df is None or merged_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, t("results", "no_data"),
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        style_no_data_axes(ax, mode)
        return ax
    shortcode_col = t("report", "shortcode")
    plot_df = merged_df.copy()
    plot_df[shortcode_col] = plot_df[shortcode_col].astype(str).str.strip()
    plot_df[shortcode_col] = plot_df[shortcode_col].str.split(r"\s*;\s*", regex=True)
    plot_df = plot_df.explode(shortcode_col)
    plot_df[shortcode_col] = plot_df[shortcode_col].astype(str).str.strip()
    plot_df = plot_df[plot_df[shortcode_col] != ""]
    if plot_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, t("results", "no_data"),
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        style_no_data_axes(ax, mode)
        return ax
    analysis_plot = (
        plot_df.groupby(shortcode_col)
        .agg(Anzahl=(shortcode_col, "count"))
        .reset_index()
        .plot(
            kind="bar",
            x=shortcode_col,
            y="Anzahl",
            alpha=1,
            rot=0,
            width=0.55,
            color=primary_color(mode),
        )
    )
    analysis_plot.set_xlabel(t("report", "shortcode"))
    plt.setp(analysis_plot.get_xticklabels(), rotation=45, ha="right")
    analysis_plot.set_ylabel(t("report", "quantity"))
    legend = analysis_plot.get_legend()
    if legend is not None:
        legend.remove()
    for container in analysis_plot.containers:
        analysis_plot.bar_label(container, label_type="edge")
    round_bar_corners(analysis_plot)
    apply_axes_style(analysis_plot, mode)
    return analysis_plot


def build_sim_plot(stats_df, teacher_name, t, mode="light"):
    """Speaker-words distribution. Returns a matplotlib axes or None."""
    if stats_df is None or len(stats_df) == 0:
        return None
    distribution = stats_df.plot(
        kind="bar",
        x="Sprecher",
        y="Gesamt_Woerter",
        alpha=1,
        rot=0,
        width=0.55,
        color=primary_color(mode),
    )
    distribution.set_xlabel(t("results", "words_total"))
    distribution.set_ylabel(t("results", "quantity"))
    legend = distribution.get_legend()
    if legend is not None:
        legend.remove()
    total = stats_df["Gesamt_Woerter"].sum() or 1
    teacher_label = t("stats", "teacher")
    students_label = t("stats", "students")
    tick_labels = [
        teacher_label if str(spk) == teacher_name else students_label
        for spk in stats_df["Sprecher"].tolist()
    ]
    distribution.set_xticks(range(len(tick_labels)))
    distribution.set_xticklabels(tick_labels)
    for container in distribution.containers:
        perc_labels = [
            f"{(bar.get_height() / total * 100):.1f}%" for bar in container
        ]
        distribution.bar_label(container, label_type="center")
        distribution.bar_label(container, labels=perc_labels, label_type="edge")
    round_bar_corners(distribution)
    apply_axes_style(distribution, mode)
    return distribution
