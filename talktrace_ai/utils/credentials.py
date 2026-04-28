"""talktrace_ai.utils.credentials

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

_KEYRING_WARNED = False


def _keyring_unavailable():
    global _KEYRING_WARNED
    if not _KEYRING_WARNED:
        _KEYRING_WARNED = True
        print("[TalkTrace] No system keyring available — API keys will not "
              "persist between sessions.", file=sys.stderr)


def safe_get_password(service, key):
    try:
        return keyring.get_password(service, key)
    except keyring.errors.NoKeyringError:
        _keyring_unavailable()
        return None
    except keyring.errors.KeyringError:
        return None
    except Exception:
        return None


def safe_set_password(service, key, value):
    try:
        keyring.set_password(service, key, value)
        return True
    except keyring.errors.NoKeyringError:
        _keyring_unavailable()
        return False
    except keyring.errors.KeyringError:
        return False
    except Exception:
        return False


def safe_delete_password(service, key):
    try:
        keyring.delete_password(service, key)
        return True
    except (keyring.errors.PasswordDeleteError,
            keyring.errors.NoKeyringError,
            keyring.errors.KeyringError):
        return False
    except Exception:
        return False


def keyring_available():
    try:
        backend = keyring.get_keyring()
    except Exception:
        return False
    name = (getattr(backend, "name", "") or backend.__class__.__name__).lower()
    # The "fail" backend is keyring's null backend used when no real backend
    # could be loaded; treat it as unavailable so the UI can warn the user.
    return "fail" not in name and "null" not in name


# Response-Cache: bei identischem (provider, model, system, user, transcript,
# codebook) liefern wir direkt die gespeicherte Antwort zurück. Spart komplette
# API-Calls z.B. beim Re-Run nach UI-Wechseln.
