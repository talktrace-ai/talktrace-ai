"""talktrace_ai.utils.file_io

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

def docx_to_json(docx_file_path):
    doc = Document(docx_file_path)

    if not doc.tables:
        text = []

        for para in doc.paragraphs:
            cleaned = para.text.strip()
            if cleaned:  # Skip empty lines
                text.append(cleaned)

        return "\n".join(text)

    table_data = []
    table = doc.tables[0]
    headers = [table.cell(0, i).text.strip() for i in range(len(table.columns))]

    for ri, row in enumerate(table.rows):
        # Row 0 is the header row; it would otherwise emit a useless
        # {"Code": "Code", "Bezeichnung": "Bezeichnung", ...} item.
        if ri == 0:
            continue
        cell_texts = [cell.text.strip() for cell in row.cells]
        row_data = {headers[i]: cell_texts[i] for i in range(len(headers))}
        table_data.append(row_data)

    return table_data




def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        txt = file.read()
    return txt


def import_file(file_dict):
    if file_dict['type'] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx_to_json(file_dict['datapath'])
    elif file_dict['type'] == "text/plain":
        return read_txt(file_dict['datapath'])
    else:
        return None




def write_txt(file_path, text):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


def write_docx_from_text(file_path, text):
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(file_path)


