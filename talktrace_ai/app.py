import re
from .myfuncs import (generate_report2, import_file, count_pupils, dialog_stats, dialog_stats_per_speaker, count_teacher_impulses,
    llm_analysis_groq, llm_analysis_openai, llm_analysis_anthropic, llm_analysis_ollama,
    get_groq_client, get_openai_client, get_anthropic_client, parse_report_impulses,
    compute_intercoder_agreement, is_valid_transcript_format, convert_to_standard_format,
    read_txt, docx_to_json, write_docx_from_text, dialog_stats_over_time,
    map_impulses_to_turn_index, code_distribution_over_time, count_transcript_turns,
    save_to_history, list_history, load_history_entry, delete_history_entry,
    DEFAULT_REPORT_SECTIONS, safe_get_password, safe_set_password, safe_delete_password,
    keyring_available, export_testing_agreement, _parse_turns)
from .transcript_analyzer import (
    analyze_transcript,
    suggest_default_options,
    convert_with_options,
    ConversionOptions,
)
from .examples.demo import (
    DEMO_TRANSCRIPT, DEMO_TEACHER_NAME, DEMO_GROUP_ID, DEMO_NUM_PUPILS,
    DEMO_CODE_LEGEND, build_demo_llm_analysis_df,
)
from .config.config_manager import ConfigManager
from .localization.translation import TRANSLATIONS
from .paths import _WELCOME_FLAG_FILE, _welcome_shown, _mark_welcome_shown, resource_path
from .ui.sidebar import build_sidebar
from .ui.head import head_content
from .ui.analysis_tab import build_analysis_tab
from .ui.results_tab import build_results_tab
from .ui.testing_tab import build_testing_tab
from .ui.options_tab import build_options_tab
from .ui.autopilot_tab import build_autopilot_tab
from .state import build_app_state
from .handlers import server_body

from pathlib import Path
import sys
import os
import webbrowser
from shiny import App, render, ui, reactive, req
from shiny._main import run_app

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from faicons import icon_svg
# Schwere Provider-SDKs (groq/openai/anthropic) werden lazy in run_analysis()
# importiert bzw. via get_*_client() aus myfuncs.py bezogen. Das senkt die
# Startzeit der App, da die SDKs nur bei tatsächlichem Gebrauch geladen werden.
import json
from datetime import date
import tempfile
import pickle
import tiktoken
import subprocess
import urllib.request
import urllib.error
import asyncio


# Note: the browser tab is opened by the launcher (start.bat) so this module
# does not trigger a second tab at import time. Kept the `webbrowser` import
# in case downstream code wants to reuse it.
url = "http://127.0.0.1:8000"

# Define the Layout

app_ui = ui.page_sidebar(
    build_sidebar(),
    head_content(),
    ui.output_ui("tt_quickstart_panel"),
    ui.output_ui("tt_demo_button_top"),
    ui.navset_tab(
        build_analysis_tab(),
        build_results_tab(),
        build_testing_tab(),
        build_autopilot_tab(),
        build_options_tab(),
        id="main_tabs",
    ),
    ui.include_css(str(resource_path("static/styles.css"))),
    title=ui.tags.span(
        "TalkTrace AI neo",
        ui.input_dark_mode(id="dark_mode"),
        class_="ttai-title",
    ),
    fillable=True,
)


def server(input, output, session):
    state = build_app_state(input, output, session)
    server_body.register(state)

# -----------------------------------------------------------------------------------------------------------   

# App als globales Objekt initiasieren, damit der server zugreifen kann
app = App(app_ui, server, debug=False)


def _find_free_port(host: str, start: int = 8000, max_tries: int = 50) -> int:
    import socket
    for offset in range(max_tries):
        candidate = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError(f"No free port found in range {start}..{start + max_tries - 1}")


def main(open_window: bool = True):
    host = "127.0.0.1"
    port = _find_free_port(host, 8000)
    print(f"[TalkTrace] Serving on http://{host}:{port}")

    if not open_window:
        run_app(app, host=host, port=port, launch_browser=False)
        return

    import threading
    import time
    import socket

    # pywebview hat plattformspezifische GUI-Abhängigkeiten (GTK/Qt auf Linux,
    # Cocoa auf macOS). Wenn der Import fehlschlägt — typisch auf Linux ohne
    # WebKit-Bindings — fallen wir auf den Standardbrowser zurück, statt zu
    # crashen.
    try:
        import webview
    except ImportError as e:
        print(f"[TalkTrace] pywebview unavailable ({e}); opening in default browser.")
        url = f"http://{host}:{port}"
        threading.Thread(
            target=lambda: (time.sleep(1.5), webbrowser.open(url)),
            daemon=True,
        ).start()
        run_app(app, host=host, port=port, launch_browser=False)
        return

    def _serve():
        run_app(app, host=host, port=port, launch_browser=False)

    threading.Thread(target=_serve, daemon=True).start()

    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)

    # Downloads im WebView-Fenster erlauben (sonst passiert beim Klick
    # auf "Report herunterladen" bzw. "Sitzung exportieren" nichts).
    webview.settings['ALLOW_DOWNLOADS'] = True

    window = webview.create_window(
        "TalkTrace AI neo",
        f"http://{host}:{port}",
        width=1280,
        height=860,
    )

    # Fenster nach kurzer Verzögerung maximieren (Fullscreen windowed),
    # da maximize() erst funktioniert, nachdem das Window initialisiert ist.
    def _maximize_window():
        time.sleep(1)
        try:
            window.maximize()
        except Exception:
            pass
    threading.Thread(target=_maximize_window, daemon=True).start()

    try:
        webview.start()
    except Exception as e:
        # WebView-Backend nicht verfügbar (z.B. Linux ohne GTK/WebKit oder
        # macOS ohne pyobjc-Frameworks): Fallback auf Browser.
        print(f"[TalkTrace] webview.start() failed ({e}); opening in default browser.")
        webbrowser.open(f"http://{host}:{port}")
        # Server läuft im Hintergrund-Thread weiter; blockierend warten.
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
