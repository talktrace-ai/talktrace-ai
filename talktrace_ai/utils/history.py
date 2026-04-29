"""talktrace_ai.utils.history"""
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

HISTORY_MAX_ENTRIES = 10
_HISTORY_DIR = Path(__file__).parent / "history"
_HISTORY_INDEX = _HISTORY_DIR / "index.json"


def _history_dir():
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return _HISTORY_DIR


def _read_history_index():
    _history_dir()
    if not _HISTORY_INDEX.exists():
        return []
    try:
        with open(_HISTORY_INDEX, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_history_index(entries):
    _history_dir()
    with open(_HISTORY_INDEX, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _safe_filename_part(s):
    s = str(s) if s is not None else ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "x"


def list_history():
    """Return the history index list, newest first."""
    return _read_history_index()


def save_to_history(session_data, group_id, model, n_turns,
                    n_pupils=None, participation_rate=None, language=None):
    """Pickle the session, append to index, evict oldest beyond HISTORY_MAX_ENTRIES."""
    _history_dir()
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    fname = f"{timestamp}_{_safe_filename_part(group_id)}_{_safe_filename_part(model)}.pkl"
    path = _HISTORY_DIR / fname
    with open(path, "wb") as f:
        pickle.dump(session_data, f)

    entry = {
        "filename": fname,
        "saved_at": now.isoformat(timespec="seconds"),
        "group_id": str(group_id) if group_id is not None else "",
        "model": str(model) if model is not None else "",
        "n_turns": int(n_turns) if n_turns is not None else 0,
        "n_pupils": int(n_pupils) if n_pupils is not None else None,
        "participation_rate": str(participation_rate) if participation_rate is not None else "",
        "language": language or "",
    }
    entries = _read_history_index()
    entries.insert(0, entry)
    if len(entries) > HISTORY_MAX_ENTRIES:
        for old in entries[HISTORY_MAX_ENTRIES:]:
            try:
                (_HISTORY_DIR / old["filename"]).unlink(missing_ok=True)
            except OSError:
                pass
        entries = entries[:HISTORY_MAX_ENTRIES]
    _write_history_index(entries)
    return entry


def load_history_entry(filename):
    path = _HISTORY_DIR / filename
    with open(path, "rb") as f:
        return pickle.load(f)


def delete_history_entry(filename):
    path = _HISTORY_DIR / filename
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    entries = [e for e in _read_history_index() if e.get("filename") != filename]
    _write_history_index(entries)
    return entries



