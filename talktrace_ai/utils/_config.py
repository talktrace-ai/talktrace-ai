"""talktrace_ai.utils._config"""
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

# Config-Singleton: ConfigManager wurde zuvor bei jedem translate()-Aufruf neu
# instanziert und las die Datei komplett vom Disk. Singleton cached die Instanz,
# aber re-liest bei mtime-Änderung — sonst bleibt bei Sprachwechsel (die app.py
# in einer anderen Instanz schreibt) hier die alte Sprache stehen.
import os as _os
_config_singleton = None
_config_mtime = 0.0

def _get_config():
    global _config_singleton, _config_mtime
    if _config_singleton is None:
        _config_singleton = ConfigManager()
        try:
            _config_mtime = _os.path.getmtime(_config_singleton.config_file)
        except OSError:
            _config_mtime = 0.0
        return _config_singleton
    try:
        mtime = _os.path.getmtime(_config_singleton.config_file)
        if mtime > _config_mtime:
            _config_singleton.config.read(_config_singleton.config_file, encoding='utf-8')
            _config_mtime = mtime
    except OSError:
        pass
    return _config_singleton


# Helper function to get translated text
def translate(section, key):
    return TRANSLATIONS[_get_config().get_localization()["current_language"]][section][key]


# Keyring-Wrapper: gracefully handle environments without a system keyring
# backend (typical on headless Linux without GNOME-Keyring/KWallet/SecretService).
# Returning None / False instead of raising lets the UI keep API keys for the
# current session only, with a non-fatal warning.
