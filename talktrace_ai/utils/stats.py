"""talktrace_ai.utils.stats

Extracted from the legacy monolithic talktrace_ai/myfuncs.py.
"""
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import json
import re
import hashlib
import pickle
import keyring
import keyring.errors
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from groq import BadRequestError, AuthenticationError, RateLimitError, InternalServerError, APIError
from openai import OpenAI
import anthropic as anthropic_sdk
from ollama import Client as OllamaClient, ResponseError as OllamaResponseError
import tiktoken
from ..localization.translation import TRANSLATIONS
from ..config.config_manager import ConfigManager

from ._config import translate

def count_pupils(transcript):
    teacher = translate("analysis", "name_teacher_var")
    sprecher_pattern = r'\b(' + re.escape(teacher) + r'|S\d{2})\b(?=:)'
    sprecher_liste = re.findall(sprecher_pattern, transcript)
    # Every distinct S\d{2} speaker counts as a student, regardless of whether
    # a teacher label is present. This supports pure student-dialog transcripts.
    sprecher_ohne_lehrer = {s for s in set(sprecher_liste) if s != teacher}
    return len(sprecher_ohne_lehrer)


def dialog_stats_per_speaker(transcript, lehrperson):
    """Per-speaker stats WITHOUT aggregating students.

    Returns a DataFrame with one row per distinct speaker label
    (teacher + each S## student), columns: Sprecher, Anzahl_Beitraege,
    Gesamt_Woerter, Durchschnitt_Woerter, Median_Woerter.
    """
    text_split = re.sub(r"//(.*?)//", r"\n\1\n", transcript, flags=re.DOTALL)
    beitrag_pattern = re.compile(rf"\b({lehrperson}|S\d{{2}})\b:\s*(.*)", re.IGNORECASE)
    beitraege = beitrag_pattern.findall(text_split)
    df = pd.DataFrame(beitraege, columns=["Sprecher", "Beitrag"])
    # Normalize teacher label to canonical case so downstream "==" comparisons work.
    df["Sprecher"] = df["Sprecher"].apply(
        lambda s: lehrperson if s.lower() == lehrperson.lower() else s
    )
    df['Wortanzahl'] = df['Beitrag'].str.split().apply(len)
    df_summary = df.groupby('Sprecher').agg(
        Anzahl_Beitraege=('Beitrag', 'count'),
        Gesamt_Woerter=('Wortanzahl', 'sum'),
        Durchschnitt_Woerter=('Wortanzahl', 'mean'),
        Median_Woerter=('Wortanzahl', 'median')
    ).reset_index()
    return df_summary


def dialog_stats(transcript, lehrperson):
    # 1. Einschübe aufsplitten
    text_split = re.sub(r"//(.*?)//", r"\n\1\n", transcript, flags=re.DOTALL)
    # 2. Regex für Beiträge
    beitrag_pattern = re.compile(rf"\b({lehrperson}|S\d{{2}})\b:\s*(.*)", re.IGNORECASE)
    beitraege = beitrag_pattern.findall(text_split)

    df = pd.DataFrame(beitraege, columns=["Sprecher", "Beitrag"])
    # Normalize teacher label to canonical case so the df_lehrer/df_schueler
    # split below (uses "==" / "!=") catches teacher rows regardless of how the
    # transcript actually spelled the name.
    df["Sprecher"] = df["Sprecher"].apply(
        lambda s: lehrperson if s.lower() == lehrperson.lower() else s
    )
    df['Wortanzahl'] = df['Beitrag'].str.split().apply(len)

    df_summary = df.groupby('Sprecher').agg(
        Anzahl_Beitraege=('Beitrag', 'count'),
        Gesamt_Woerter=('Wortanzahl', 'sum'),
        Durchschnitt_Woerter=('Wortanzahl', 'mean'),
        Median_Woerter=('Wortanzahl', 'median')
    ).reset_index()

    # Schüler vs. Lehrer trennen
    df_lehrer = df_summary[df_summary['Sprecher'] == lehrperson]
    df_schueler = df_summary[df_summary['Sprecher'] != lehrperson]

    schueler_beitraege = df_schueler['Anzahl_Beitraege'].sum()
    schueler_summary = pd.DataFrame({
        'Sprecher': ['Schüler:innen'],
        'Anzahl_Beitraege': [schueler_beitraege],
        'Gesamt_Woerter': [df_schueler['Gesamt_Woerter'].sum()],
        'Durchschnitt_Woerter': [df_schueler['Gesamt_Woerter'].sum() / schueler_beitraege if schueler_beitraege > 0 else 0],
        'Median_Woerter': [df_schueler['Median_Woerter'].median()]
    })

    return pd.concat([df_lehrer, schueler_summary], ignore_index=True)


def _parse_turns(transcript, lehrperson):
    """Ordered list of (speaker, utterance) tuples — same parsing as dialog_stats.
    Teacher label is normalized to the canonical `lehrperson` casing so callers
    can rely on `spk == lehrperson` comparisons."""
    text_split = re.sub(r"//(.*?)//", r"\n\1\n", transcript, flags=re.DOTALL)
    beitrag_pattern = re.compile(rf"\b({lehrperson}|S\d{{2}})\b:\s*(.*)", re.IGNORECASE)
    matches = beitrag_pattern.findall(text_split)
    teacher_lower = lehrperson.lower()
    return [
        (lehrperson if spk.lower() == teacher_lower else spk, utt)
        for spk, utt in matches
    ]


def dialog_stats_over_time(transcript, lehrperson, n_segments=3, segment_labels=None):
    """Teacher vs. students word/turn counts per equal-sized transcript segment.

    Long-format DataFrame: Abschnitt, Sprecher_Gruppe, Wörter, Beiträge.
    """
    cols = ["Abschnitt", "Sprecher_Gruppe", "Wörter", "Beiträge"]
    beitraege = _parse_turns(transcript, lehrperson)
    if not beitraege:
        return pd.DataFrame(columns=cols)

    n = len(beitraege)
    n_segments = max(1, int(n_segments))
    bucket_size = max(1, n // n_segments)

    teacher_label = translate("stats", "teacher")
    students_label = translate("stats", "students")

    if segment_labels is None or len(segment_labels) != n_segments:
        segment_labels = [str(i + 1) for i in range(n_segments)]

    rows = []
    for seg_idx in range(n_segments):
        start = seg_idx * bucket_size
        end = n if seg_idx == n_segments - 1 else min(start + bucket_size, n)
        slice_ = beitraege[start:end] if start < n else []
        teacher_words = sum(len(u.split()) for spk, u in slice_ if spk == lehrperson)
        teacher_turns = sum(1 for spk, _ in slice_ if spk == lehrperson)
        student_words = sum(len(u.split()) for spk, u in slice_ if spk != lehrperson)
        student_turns = sum(1 for spk, _ in slice_ if spk != lehrperson)
        seg_label = segment_labels[seg_idx]
        rows.append({"Abschnitt": seg_label, "Sprecher_Gruppe": teacher_label,
                     "Wörter": teacher_words, "Beiträge": teacher_turns})
        rows.append({"Abschnitt": seg_label, "Sprecher_Gruppe": students_label,
                     "Wörter": student_words, "Beiträge": student_turns})
    return pd.DataFrame(rows, columns=cols)


def map_impulses_to_turn_index(analysis_df, transcript, lehrperson):
    """Backfill a turn_index column on a coded-impulse DataFrame.

    The LLM output has no direct link to transcript position; match each
    impulse text to the earliest still-unused turn (monotonic), with a
    one-shot fallback before the cursor for out-of-order LLM output.
    Unmatched impulses get None.
    """
    if analysis_df is None or analysis_df.empty or "Impuls" not in analysis_df.columns:
        out = (analysis_df.copy() if analysis_df is not None
               else pd.DataFrame(columns=["Impuls"]))
        out["turn_index"] = None
        return out

    turns = _parse_turns(transcript, lehrperson)

    def normalize(s):
        return re.sub(r"\s+", " ", str(s).strip().lower())

    norm_turns = [normalize(utt) for _, utt in turns]
    used = [False] * len(turns)
    indices = []
    cursor = 0

    for impuls in analysis_df["Impuls"].tolist():
        target = normalize(impuls)
        match = None
        if target:
            for j in range(cursor, len(turns)):
                if used[j]:
                    continue
                tn = norm_turns[j]
                if target == tn or target in tn or tn in target:
                    match = j
                    break
            if match is None:
                for j in range(0, cursor):
                    if used[j]:
                        continue
                    tn = norm_turns[j]
                    if target == tn or target in tn or tn in target:
                        match = j
                        break
        if match is not None:
            used[match] = True
            indices.append(match)
            cursor = match + 1
        else:
            indices.append(None)

    out = analysis_df.copy()
    out["turn_index"] = indices
    return out


def code_distribution_over_time(analysis_df_with_index, total_turns, n_segments=3, segment_labels=None):
    """Per-segment share of each shortcode (Anteil sums to ~1 per segment)."""
    cols = ["Abschnitt", "Shortcode", "Anteil", "Anzahl"]
    if (analysis_df_with_index is None or analysis_df_with_index.empty
            or total_turns <= 0
            or "turn_index" not in analysis_df_with_index.columns
            or "Shortcode" not in analysis_df_with_index.columns):
        return pd.DataFrame(columns=cols)

    n_segments = max(1, int(n_segments))
    if segment_labels is None or len(segment_labels) != n_segments:
        segment_labels = [str(i + 1) for i in range(n_segments)]

    bucket_size = max(1, total_turns // n_segments)

    df = analysis_df_with_index.dropna(subset=["turn_index"]).copy()
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["turn_index"] = df["turn_index"].astype(int)
    df["_bucket"] = (df["turn_index"] // bucket_size).clip(upper=n_segments - 1)

    rows = []
    for seg_idx in range(n_segments):
        seg_df = df[df["_bucket"] == seg_idx]
        seg_codes = seg_df["Shortcode"].astype(str).str.strip()
        seg_codes = seg_codes[seg_codes != ""]
        total = len(seg_codes)
        if total == 0:
            continue
        seg_label = segment_labels[seg_idx]
        for code, count in seg_codes.value_counts().items():
            rows.append({
                "Abschnitt": seg_label,
                "Shortcode": code,
                "Anteil": count / total,
                "Anzahl": int(count),
            })
    return pd.DataFrame(rows, columns=cols)


def count_transcript_turns(transcript, lehrperson):
    """Count the parsed turns in a transcript (teacher + all S## speakers)."""
    return len(_parse_turns(transcript, lehrperson))




def count_teacher_impulses(df, teacher_name):
    matches = df.loc[df['Sprecher'] == teacher_name, 'Anzahl_Beitraege']
    if matches.empty:
        return 0
    return matches.values[0]

