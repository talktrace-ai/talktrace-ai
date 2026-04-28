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
]
