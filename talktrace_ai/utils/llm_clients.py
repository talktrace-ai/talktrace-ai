"""talktrace_ai.utils.llm_clients"""
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

# Client-Cache: je API-Key wird nur ein SDK-Client instanziert.
_client_cache = {}


def get_groq_client(api_key):
    from groq import Groq
    key = ("groq", api_key)
    if key not in _client_cache:
        _client_cache[key] = Groq(api_key=api_key)
    return _client_cache[key]


def get_openai_client(api_key):
    key = ("openai", api_key)
    if key not in _client_cache:
        _client_cache[key] = OpenAI(api_key=api_key)
    return _client_cache[key]


def get_anthropic_client(api_key):
    key = ("anthropic", api_key)
    if key not in _client_cache:
        _client_cache[key] = anthropic_sdk.Anthropic(api_key=api_key)
    return _client_cache[key]


