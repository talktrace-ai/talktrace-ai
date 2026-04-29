"""Pure (non-reactive) helpers used by the Autopilot orchestrator.

The manual analysis flow in ``handlers/sidebar/_analysis.py`` reads its
provider/model/switch values from Shiny reactive inputs and global state.
The Autopilot needs to run two codings back-to-back with explicit overrides
without touching that reactive state, so the prompt-building and LLM-call
logic is mirrored here as plain functions:

- ``build_effective_prompts`` ports the speaker-filter + multi-coding
  suffixing + sanitizing logic from ``handlers/sidebar/_prompts.py``.
- ``run_llm_coding_once`` ports the non-streaming LLM-call branch from
  ``handlers/sidebar/_analysis.py``: dispatch by provider, parse the JSON
  envelope, return a DataFrame.

The functions return data; they never call ``reactive.set`` or ``reactive.lock``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import pandas as pd

from .anthropic import llm_analysis_anthropic
from .groq import llm_analysis_groq
from .openai import llm_analysis_openai
from .ollama import llm_analysis_ollama


def _sanitize_prompt_for_speakers(text: str, teacher: bool, students: bool) -> str:
    if teacher and students:
        return text
    if not teacher:
        text = text.replace("Lehrperson UND Schüler:innen", "Schüler:innen")
        text = text.replace("Lehrperson und Schüler:innen", "Schüler:innen")
        text = text.replace("ALLER Sprecher:innen (Lehrperson UND Schüler:innen)", "der Schüler:innen")
        text = text.replace("ALLER Sprecher:innen", "der Schüler:innen")
        text = text.replace("Lehrperson", "")
        text = text.replace("LEHRER", "")
        text = text.replace("Lehrer", "")
    if not students:
        text = text.replace("Lehrperson UND Schüler:innen", "Lehrperson")
        text = text.replace("Lehrperson und Schüler:innen", "Lehrperson")
        text = text.replace("ALLER Sprecher:innen (Lehrperson UND Schüler:innen)", "der Lehrperson")
        text = text.replace("ALLER Sprecher:innen", "der Lehrperson")
        text = text.replace("Schüler:innen", "")
        text = text.replace("S01, S02, S03…", "")
        text = text.replace("S01, S02, S03...", "")
        text = text.replace("S01, S02, S03", "")
        text = text.replace("S01", "")
    if not teacher:
        text = text.replace("teacher AND students", "students")
        text = text.replace("teacher and students", "students")
        text = text.replace("ALL speakers (teacher AND students)", "students")
        text = text.replace("ALL speakers", "students")
        text = text.replace("teacher", "")
    if not students:
        text = text.replace("teacher AND students", "teacher")
        text = text.replace("teacher and students", "teacher")
        text = text.replace("ALL speakers (teacher AND students)", "teacher")
        text = text.replace("ALL speakers", "teacher")
        text = text.replace("students", "")
        text = text.replace("S01, S02, S03…", "")
        text = text.replace("S01, S02, S03...", "")
        text = text.replace("S01, S02, S03", "")
        text = text.replace("S01", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text.strip()


def _speaker_filter_suffix(t, *, kind: str, teacher: bool, students: bool, teacher_name: str) -> str:
    prefix = "user_prompt_filter" if kind == "user" else "prompt_filter"
    if teacher and students:
        return ""
    if teacher and not students:
        return t("sidebar", f"{prefix}_teacher_only").format(teacher_name=teacher_name)
    if students and not teacher:
        return t("sidebar", f"{prefix}_students_only").format(teacher_name=teacher_name)
    return t("sidebar", f"{prefix}_none")


def _multi_coding_suffix(t, *, kind: str, multi_coding: bool) -> str:
    prefix = "user_prompt_multi_coding" if kind == "user" else "prompt_multi_coding"
    key = f"{prefix}_{'on' if multi_coding else 'off'}"
    return t("sidebar", key)


def build_effective_prompts(
    base_system: str,
    base_user: str,
    *,
    t,
    teacher_on: bool,
    students_on: bool,
    multi_coding: bool,
    teacher_name: str,
):
    """Return (system_prompt, user_prompt) with speaker + multi-coding instructions.

    Mirrors ``effective_system_prompt`` / ``effective_user_prompt`` from
    ``handlers/sidebar/_prompts.py``, but parametric instead of reactive.
    """
    sys_base = _sanitize_prompt_for_speakers(base_system, teacher_on, students_on)
    sys_prompt = (
        sys_base
        + _speaker_filter_suffix(t, kind="system", teacher=teacher_on, students=students_on, teacher_name=teacher_name)
        + _multi_coding_suffix(t, kind="system", multi_coding=multi_coding)
    )

    user_raw = _sanitize_prompt_for_speakers(base_user, teacher_on, students_on)
    combined = (
        _speaker_filter_suffix(t, kind="user", teacher=teacher_on, students=students_on, teacher_name=teacher_name)
        + _multi_coding_suffix(t, kind="user", multi_coding=multi_coding)
    )
    if not combined:
        user_prompt = user_raw
    elif "{transcript}" in user_raw:
        target = "{transcript}"
        idx = user_raw.index(target)
        insert_pos = idx + len(target)
        user_prompt = user_raw[:insert_pos] + "\n\n" + combined + user_raw[insert_pos:]
    else:
        user_prompt = user_raw + combined
    return sys_prompt, user_prompt


def _build_client(provider: str, *, api_key: Optional[str]):
    """Lazy-import provider SDK clients so startup stays fast."""
    if provider == "openai":
        from ..llm_clients import get_openai_client
        return get_openai_client(api_key)
    if provider == "groq":
        from ..llm_clients import get_groq_client
        return get_groq_client(api_key)
    if provider == "anthropic":
        from ..llm_clients import get_anthropic_client
        return get_anthropic_client(api_key)
    return None


def run_llm_coding_once(
    *,
    provider: str,
    model: str,
    transcript: Any,
    codebook: Any,
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
):
    """Run a single non-streaming LLM coding pass and return (df, raw_json, err).

    On error, df is None and err is a human-readable string describing what
    went wrong; raw_json may still hold the unparsed provider response. This
    is the function the Autopilot calls twice in sequence.
    """
    if provider not in {"openai", "groq", "anthropic", "ollama"}:
        return None, None, f"Unsupported provider: {provider}"

    if provider != "ollama" and not api_key:
        return None, None, f"No API key configured for provider {provider}"

    try:
        if provider == "openai":
            client = _build_client(provider, api_key=api_key)
            raw = llm_analysis_openai(system_prompt, user_prompt, model, transcript, codebook, client)
        elif provider == "groq":
            client = _build_client(provider, api_key=api_key)
            raw = llm_analysis_groq(system_prompt, user_prompt, model, transcript, codebook, client)
        elif provider == "anthropic":
            client = _build_client(provider, api_key=api_key)
            raw = llm_analysis_anthropic(system_prompt, user_prompt, model, transcript, codebook, client)
        else:  # ollama
            raw = llm_analysis_ollama(system_prompt, user_prompt, model, transcript, codebook)
    except Exception as exc:
        return None, None, f"{provider} call failed: {exc}"

    if raw is None:
        return None, None, "No response received from provider"

    if isinstance(raw, str) and '"error":' in raw:
        try:
            return None, raw, json.loads(raw).get("error", "unknown error")
        except Exception:
            return None, raw, "unknown error"

    try:
        new_data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        return None, raw, f"Could not parse response JSON: {exc}"

    if isinstance(new_data, list):
        new_data = {"analysis": new_data}
    analysis_items = new_data.get("analysis", []) if isinstance(new_data, dict) else []
    if not isinstance(analysis_items, list):
        analysis_items = []
    if len(analysis_items) == 0:
        return None, raw, "LLM returned 0 coded items"

    for item in analysis_items:
        if isinstance(item, dict) and "Sprecher" not in item:
            item["Sprecher"] = ""

    df = pd.DataFrame(analysis_items, columns=["#", "Sprecher", "Shortcode", "Impuls"])
    return df, raw, None
