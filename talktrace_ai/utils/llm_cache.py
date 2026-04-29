"""talktrace_ai.utils.llm_cache"""
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

_RESPONSE_CACHE_MAX = 32
_response_cache: "OrderedDict[str, str]" = OrderedDict()


def _cache_key(provider, model, system_prompt, user_prompt, transcript, codebook, extra=""):
    # Lazy import: llm_analysis._json lives in a sibling subpackage whose
    # provider modules import back from this file. Importing at module load
    # would create a cycle (llm_cache → llm_analysis.__init__ → groq.py →
    # llm_cache); deferring to call time breaks it.
    from .llm_analysis._json import _format_codebook
    h = hashlib.md5()
    for part in (provider, model, system_prompt, user_prompt, str(transcript),
                 _format_codebook(codebook), extra):
        h.update(part.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_get(key):
    if key in _response_cache:
        _response_cache.move_to_end(key)
        return _response_cache[key]
    return None


def _cache_put(key, value):
    # Fehlerantworten nicht cachen.
    if not value or '"error":' in value:
        return
    # Leere Codierungen nicht cachen — sonst bleiben fehlerhafte
    # API-Calls hängen und jeder Retry liefert dasselbe leere Ergebnis.
    if '"analysis": []' in value or '"analysis":[]' in value:
        return
    _response_cache[key] = value
    _response_cache.move_to_end(key)
    while len(_response_cache) > _RESPONSE_CACHE_MAX:
        _response_cache.popitem(last=False)


