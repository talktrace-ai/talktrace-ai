"""Shared imports for talktrace_ai.handlers.* modules.

Each handler does `from ._common import *` to get the same import surface
that the original monolithic app.py had. This is a deliberate broad import
to keep the handler bodies textually identical to their pre-refactor form.
"""
import re
import os
import sys
import json
import asyncio
import tempfile
import pickle
import subprocess
import urllib.request
import urllib.error
import webbrowser
from datetime import date
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tiktoken

from shiny import ui, render, reactive, req
from faicons import icon_svg

from ..myfuncs import (
    generate_report2, import_file, count_pupils, dialog_stats, dialog_stats_per_speaker, count_teacher_impulses,
    llm_analysis_groq, llm_analysis_openai, llm_analysis_anthropic, llm_analysis_ollama,
    llm_analysis_groq_stream, llm_analysis_openai_stream,
    llm_analysis_anthropic_stream, llm_analysis_ollama_stream,
    async_stream,
    get_groq_client, get_openai_client, get_anthropic_client, parse_report_impulses,
    compute_intercoder_agreement, is_valid_transcript_format, convert_to_standard_format,
    read_txt, docx_to_json, write_docx_from_text, dialog_stats_over_time,
    map_impulses_to_turn_index, code_distribution_over_time, count_transcript_turns,
    save_to_history, list_history, load_history_entry, delete_history_entry,
    DEFAULT_REPORT_SECTIONS, safe_get_password, safe_set_password, safe_delete_password,
    keyring_available, export_testing_agreement, export_testing_agreement_any,
    _parse_turns,
)
from ..transcript_analyzer import (
    analyze_transcript,
    suggest_default_options,
    convert_with_options,
    ConversionOptions,
)
from ..examples.demo import (
    DEMO_TRANSCRIPT, DEMO_TEACHER_NAME, DEMO_GROUP_ID, DEMO_NUM_PUPILS,
    DEMO_CODE_LEGEND, DEMO_CODEBOOK, build_demo_llm_analysis_df,
)
from ..config.config_manager import ConfigManager
from ..localization.translation import TRANSLATIONS
from ..paths import _WELCOME_FLAG_FILE, _welcome_shown, _mark_welcome_shown, resource_path


def detect_transcript_format_status(file_dict, teacher=None):
    """Return one of "valid" | "invalid" | "unsupported" for an uploaded
    transcript file dict, or None if no file was given.

    "unsupported" covers PDF and .docx files that contain tables (which the
    current pipeline can't read as plain transcript). The wizard button will
    surface those cases via its existing modal. "invalid" means the file is
    readable but doesn't match the S01/TEACHER turn-per-line format and
    therefore should be run through the conversion wizard.
    """
    if not file_dict:
        return None
    name = file_dict.get("name", "")
    ext = os.path.splitext(name)[1].lower()
    datapath = file_dict.get("datapath")
    if not datapath:
        return None
    if ext == ".pdf":
        return "unsupported"
    try:
        if ext == ".docx":
            content = docx_to_json(datapath)
            if not isinstance(content, str):
                return "unsupported"
            text = content
        else:
            text = read_txt(datapath)
    except Exception:
        return "unsupported"
    return "valid" if is_valid_transcript_format(text, teacher) else "invalid"


# Substrings (matched after normalization: lowercase, umlaut-folded, no
# non-letters). The substring approach automatically covers compounds like
# "Klassenlehrer", "Fachlehrerin", "Lehrkräfte" as well as gendered spellings
# such as "Lehrer*in", "Lehrer:in", "Lehrer/in" which all collapse to
# "lehrerin" once non-letters are stripped.
_TEACHER_LABEL_KEYWORDS = (
    # English
    "teacher", "instructor", "tutor", "educator", "professor",
    "lecturer", "trainer", "mentor", "faculty",
    # German (incl. neutral/inclusive forms via substring matching)
    "lehrer", "lehrerin", "lehrkraft", "lehrkraef", "lehrperson", "lehrende",
    "paedagog", "erzieher", "ausbilder", "dozent", "referent",
    "schulleit", "klassenleit", "kursleit",
)


def _fold_label(s: str) -> str:
    s = s.lower()
    s = (s.replace("ä", "ae").replace("ö", "oe")
         .replace("ü", "ue").replace("ß", "ss"))
    return re.sub(r'[^a-z]', '', s)


def detect_teacher_label(file_dict):
    """Scan the uploaded transcript for a speaker label that looks like a
    teacher (Lehrer/Teacher/...). Returns the label as it appears in the
    transcript, or None if no obvious teacher label is found.

    Conservative on purpose: only matches keyword-based labels, not single
    letters like "L:" — those are ambiguous and would risk false positives.
    """
    if not file_dict:
        return None
    name = file_dict.get("name", "")
    ext = os.path.splitext(name)[1].lower()
    datapath = file_dict.get("datapath")
    if not datapath or ext == ".pdf":
        return None
    try:
        if ext == ".docx":
            content = docx_to_json(datapath)
            if not isinstance(content, str):
                return None
            text = content
        else:
            text = read_txt(datapath)
    except Exception:
        return None

    try:
        analysis = analyze_transcript(text)
    except Exception:
        return None
    for raw in analysis.speakers:
        norm = _fold_label(raw)
        if any(kw in norm for kw in _TEACHER_LABEL_KEYWORDS):
            return raw
    return None


def main_tab_is(value, slot_id: str) -> bool:
    """Robust check whether the navset's current value points at a given
    tab title slot (e.g. "loc_title_results").

    The value of ``input.main_tabs()`` is the rendered HTML of the title
    output slot — its CSS class differs depending on whether the slot is
    rendered as @render.text (``shiny-text-output``) or @render.ui
    (``shiny-html-output``). Substring-matching the id keeps callers
    independent of that detail.
    """
    if not value:
        return False
    return f'id="{slot_id}"' in str(value)


def mark_tab_unread(badge_value, current_main_tab, slot_id: str):
    """Set a tab's badge to "unread" — but if the user is already on that
    tab, jump straight to "read" so we don't flash a red dot at them while
    they're looking at the freshly populated content."""
    if main_tab_is(current_main_tab, slot_id):
        badge_value.set("read")
    else:
        badge_value.set("unread")


def tab_title_with_badge(text, status):
    """Render a tab title with an optional notification dot.

    status: None | "unread" | "read". The dot stays visible in "read"
    state (different color) so the user keeps a subtle indicator that
    the tab carries data they've already seen.
    """
    if not status:
        return ui.span(text)
    return ui.span(
        text,
        ui.tags.span(class_=f"ttai-tab-badge {status}"),
    )


def render_transcript_format_status_ui(status, t):
    """Build the small icon shown next to the wand button. Returns None when
    nothing should be displayed (no upload yet)."""
    if not status:
        return None
    spec = {
        "valid": ("check", "#28a745", t("analysis", "format_status_valid")),
        "invalid": ("triangle-exclamation", "#f0ad4e", t("analysis", "format_status_invalid")),
        "unsupported": ("circle-xmark", "#d9534f", t("analysis", "format_status_unsupported")),
    }.get(status)
    if spec is None:
        return None
    name, color, tip = spec
    return ui.tooltip(
        ui.tags.span(
            icon_svg(name),
            style=f"color: {color}; font-size: 1.1rem; line-height: 1;",
        ),
        tip,
        placement="right",
    )

# Star-import friendliness: list every name we re-export so wildcard imports
# pick up underscore-prefixed paths helpers and similar.
__all__ = [
    "re", "os", "sys", "json", "asyncio", "tempfile", "pickle",
    "subprocess", "urllib", "webbrowser", "date", "Path",
    "pd", "matplotlib", "plt", "tiktoken",
    "ui", "render", "reactive", "req", "icon_svg",
    "generate_report2", "import_file", "count_pupils", "dialog_stats",
    "dialog_stats_per_speaker", "count_teacher_impulses",
    "llm_analysis_groq", "llm_analysis_openai", "llm_analysis_anthropic", "llm_analysis_ollama",
    "llm_analysis_groq_stream", "llm_analysis_openai_stream",
    "llm_analysis_anthropic_stream", "llm_analysis_ollama_stream",
    "async_stream",
    "get_groq_client", "get_openai_client", "get_anthropic_client",
    "parse_report_impulses", "compute_intercoder_agreement",
    "is_valid_transcript_format", "convert_to_standard_format",
    "read_txt", "docx_to_json", "write_docx_from_text",
    "dialog_stats_over_time", "map_impulses_to_turn_index",
    "code_distribution_over_time", "count_transcript_turns",
    "save_to_history", "list_history", "load_history_entry", "delete_history_entry",
    "DEFAULT_REPORT_SECTIONS",
    "safe_get_password", "safe_set_password", "safe_delete_password",
    "keyring_available", "export_testing_agreement", "export_testing_agreement_any",
    "_parse_turns",
    "analyze_transcript", "suggest_default_options",
    "convert_with_options", "ConversionOptions",
    "DEMO_TRANSCRIPT", "DEMO_TEACHER_NAME", "DEMO_GROUP_ID",
    "DEMO_NUM_PUPILS", "DEMO_CODE_LEGEND", "DEMO_CODEBOOK", "build_demo_llm_analysis_df",
    "ConfigManager", "TRANSLATIONS",
    "_WELCOME_FLAG_FILE", "_welcome_shown", "_mark_welcome_shown", "resource_path",
    "detect_transcript_format_status",
    "render_transcript_format_status_ui",
    "detect_teacher_label",
    "main_tab_is",
    "tab_title_with_badge",
    "mark_tab_unread",
]
