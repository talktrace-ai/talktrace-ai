"""talktrace_ai.utils.transcript_format"""
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

_SPEAKER_HEADER_RE = re.compile(r'^\s*"?\s*SPEAKER[_\s-]*(\d+)\s*:?\s*"?\s*$', re.IGNORECASE)
_VALID_LINE_RE = re.compile(r'^\s*"?\s*S\d{2}\s*:')


def is_valid_transcript_format(text, teacher=None):
    """Return True if the transcript already follows the expected
    one-line-per-turn `S0X:` (or teacher) format and contains no
    `SPEAKER_XX` headers on their own lines."""
    if not text:
        return False
    for line in text.splitlines():
        if _SPEAKER_HEADER_RE.match(line):
            return False
    teacher_re = None
    if teacher:
        teacher_re = re.compile(r'^\s*"?\s*' + re.escape(teacher) + r'\s*:')
    for line in text.splitlines():
        if _VALID_LINE_RE.match(line):
            return True
        if teacher_re and teacher_re.match(line):
            return True
    return False


def convert_to_standard_format(text):
    """Convert noScribe-style transcripts (SPEAKER_XX on its own line,
    body on following lines) into one `S0X: text` line per turn.

    Speaker tags are renumbered deterministically in order of first
    appearance (SPEAKER_02 -> S01 if it appears first, etc.).
    """
    if not text:
        return ""

    lines = text.splitlines()
    turns = []
    current_speaker = None
    current_buf = []

    def flush():
        if current_speaker is not None:
            body = " ".join(s.strip() for s in current_buf if s.strip())
            body = re.sub(r"\s+", " ", body).strip()
            if body:
                turns.append((current_speaker, body))

    for raw in lines:
        m = _SPEAKER_HEADER_RE.match(raw)
        if m:
            flush()
            current_speaker = m.group(1)
            current_buf = []
            continue
        if current_speaker is None:
            continue
        current_buf.append(raw)
    flush()

    out_lines = []
    for spk_num, body in turns:
        try:
            n = int(spk_num)
        except (TypeError, ValueError):
            n = 0
        out_lines.append(f"S{n:02d}: {body}")

    return "\n".join(out_lines)


