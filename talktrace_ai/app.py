import re
from .myfuncs import generate_report2, import_file, count_pupils, dialog_stats, dialog_stats_per_speaker, count_teacher_impulses, llm_analysis_groq, llm_analysis_openai, llm_analysis_anthropic, llm_analysis_ollama, get_groq_client, get_openai_client, get_anthropic_client, parse_report_impulses, compute_intercoder_agreement
from .config.config_manager import ConfigManager
from .localization.translation import TRANSLATIONS

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
import keyring
import keyring.errors
import tiktoken
import subprocess
import urllib.request
import urllib.error
import asyncio


# Path Helper for css-files
def resource_path(relative_path: str) -> Path:
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).parent / relative_path

# Note: the browser tab is opened by the launcher (start.bat) so this module
# does not trigger a second tab at import time. Kept the `webbrowser` import
# in case downstream code wants to reuse it.
url = "http://127.0.0.1:8000"

# Define the Layout

# Sidebar Menu with Model Selection, Analysis, Report Download and Session Management
OBSIDIAN_CSS = """
/* --- Obsidian-inspired dark theme (Shiny/bslib) ------------------------ */

/* Permanently override bslib's initial --bslib-sidebar-main-bg (#f8f8f8)
   with a fully transparent value so no white bleed-through occurs.
   !important on custom properties beats bslib's later-loading definition. */
:root,
.bslib-sidebar-layout,
html .bslib-sidebar-layout,
html body .bslib-sidebar-layout {
    --bslib-sidebar-main-bg: #f8f8f800 !important;
}

/* Bootstrap 5.3 CSS variable overrides — applied when bslib sets
   data-bs-theme="dark" on <html>                                         */
html[data-bs-theme="dark"] {
    --bs-body-bg: #1e1e1e;
    --bs-body-color: #dcddde;
    --bs-border-color: #3a3a3a;
    --bs-secondary-bg: #262626;
    --bs-tertiary-bg: #2a2a2a;
    --bs-primary: #a78bfa;
    --bs-primary-rgb: 167, 139, 250;
    --bs-link-color: #c4b5fd;
    --bs-link-hover-color: #ddd6fe;
    --bs-emphasis-color: #ede9fe;
    color-scheme: dark;
}

/* ---- Page / body ---- */
/* Shiny forces body{background:transparent!important}, so the visible
   background comes from <html> itself — must be set here in dark mode. */
html[data-bs-theme="dark"] {
    background-color: #1e1e1e !important;
}
html[data-bs-theme="dark"] body,
html[data-bs-theme="dark"] .bslib-page-fill,
html[data-bs-theme="dark"] .bslib-page-sidebar {
    background-color: #1e1e1e !important;
    color: #dcddde !important;
}

/* ---- bslib layout containers ---- */
/* Neutralise bslib's built-in "background-color: var(--_main-bg)" on .main
   so it never produces an unwanted colour in light mode.                     */
.bslib-sidebar-layout {
    background-color: #f8f8f800 !important;
    --bslib-sidebar-main-bg: #f8f8f800;
}
.bslib-sidebar-layout > .main,
.bslib-page-main,
main.bslib-page-main {
    background-color: transparent !important;
}

/* Dark-mode overrides */
html[data-bs-theme="dark"] .bslib-sidebar-layout {
    --bslib-sidebar-main-bg: #1e1e1e !important;
    --_main-bg: #1e1e1e !important;
}
html[data-bs-theme="dark"] .bslib-sidebar-layout > .main,
html[data-bs-theme="dark"] .bslib-page-main,
html[data-bs-theme="dark"] main.bslib-page-main,
html[data-bs-theme="dark"] main.bslib-page-main.html-fill-container {
    background-color: #1e1e1e !important;
    color: #dcddde !important;
}

/* ---- Sidebar ---- */
html[data-bs-theme="dark"] .bslib-sidebar-layout > .sidebar,
html[data-bs-theme="dark"] .bslib-sidebar-layout aside.sidebar,
html[data-bs-theme="dark"] aside.sidebar {
    background-color: #202020 !important;
    border-right: 1px solid #2d2d2d !important;
    color: #dcddde !important;
}
html[data-bs-theme="dark"] .sidebar-content {
    background-color: #202020 !important;
}

/* ---- Cards / value boxes ---- */
html[data-bs-theme="dark"] .card,
html[data-bs-theme="dark"] .bslib-value-box,
html[data-bs-theme="dark"] .bslib-card {
    background-color: #262626 !important;
    border-color: #3a3a3a !important;
    color: #dcddde !important;
}
html[data-bs-theme="dark"] .card-header {
    background-color: #2a2a2a !important;
    border-bottom-color: #3a3a3a !important;
    color: #ede9fe !important;
}
html[data-bs-theme="dark"] .card-body {
    background-color: #262626 !important;
    color: #dcddde !important;
}

/* ---- Nav tabs ---- */
html[data-bs-theme="dark"] .nav-tabs {
    border-bottom-color: #3a3a3a !important;
}
html[data-bs-theme="dark"] .nav-tabs .nav-link {
    color: #9ca3af !important;
    background-color: transparent !important;
}
html[data-bs-theme="dark"] .nav-tabs .nav-link.active,
html[data-bs-theme="dark"] .nav-tabs .nav-link:hover {
    background-color: #2a2a2a !important;
    border-color: #3a3a3a #3a3a3a transparent !important;
    color: #a78bfa !important;
}
html[data-bs-theme="dark"] .tab-content,
html[data-bs-theme="dark"] .tab-pane {
    background-color: #1e1e1e !important;
    color: #dcddde !important;
}

/* ---- Form inputs ---- */
html[data-bs-theme="dark"] .form-control,
html[data-bs-theme="dark"] .form-select,
html[data-bs-theme="dark"] textarea,
html[data-bs-theme="dark"] input[type="text"],
html[data-bs-theme="dark"] input[type="number"],
html[data-bs-theme="dark"] input[type="password"],
html[data-bs-theme="dark"] input[type="search"] {
    background-color: #2a2a2a !important;
    border-color: #3a3a3a !important;
    color: #dcddde !important;
}
html[data-bs-theme="dark"] .form-control:focus,
html[data-bs-theme="dark"] .form-select:focus {
    background-color: #2a2a2a !important;
    border-color: #a78bfa !important;
    color: #ede9fe !important;
    box-shadow: 0 0 0 0.2rem rgba(167, 139, 250, 0.25) !important;
}
html[data-bs-theme="dark"] label,
html[data-bs-theme="dark"] .form-label {
    color: #dcddde !important;
}

/* ---- Buttons ---- */
html[data-bs-theme="dark"] .btn-primary,
html[data-bs-theme="dark"] .btn-default {
    background-color: #7c3aed !important;
    border-color: #7c3aed !important;
    color: #ffffff !important;
}
html[data-bs-theme="dark"] .btn-primary:hover,
html[data-bs-theme="dark"] .btn-default:hover {
    background-color: #8b5cf6 !important;
    border-color: #8b5cf6 !important;
}
html[data-bs-theme="dark"] .btn-secondary {
    background-color: #3a3a3a !important;
    border-color: #3a3a3a !important;
    color: #dcddde !important;
}
html[data-bs-theme="dark"] .btn-outline-primary {
    color: #a78bfa !important;
    border-color: #a78bfa !important;
    background-color: transparent !important;
}
html[data-bs-theme="dark"] .btn-outline-primary:hover {
    background-color: #a78bfa !important;
    color: #1e1e1e !important;
}
html[data-bs-theme="dark"] .btn-outline-secondary {
    color: #9ca3af !important;
    border-color: #3a3a3a !important;
    background-color: transparent !important;
}
html[data-bs-theme="dark"] .btn-light {
    background-color: #2a2a2a !important;
    border-color: #3a3a3a !important;
    color: #dcddde !important;
}

/* ---- Sidebar action buttons ---- */
html[data-bs-theme="dark"] aside.sidebar .btn,
html[data-bs-theme="dark"] .sidebar-content .btn {
    background-color: #2a2a2a !important;
    border-color: #3a3a3a !important;
    color: #dcddde !important;
}
html[data-bs-theme="dark"] aside.sidebar .btn:hover,
html[data-bs-theme="dark"] .sidebar-content .btn:hover {
    background-color: #7c3aed !important;
    border-color: #7c3aed !important;
    color: #ffffff !important;
}

/* ---- Semantic buttons (success / danger) ---- */
html[data-bs-theme="dark"] .btn-success,
html[data-bs-theme="dark"] aside.sidebar .btn-success,
html[data-bs-theme="dark"] .sidebar-content .btn-success {
    background-color: #2f9e44 !important;
    border-color: #2f9e44 !important;
    color: #ffffff !important;
}
html[data-bs-theme="dark"] .btn-success:hover,
html[data-bs-theme="dark"] aside.sidebar .btn-success:hover,
html[data-bs-theme="dark"] .sidebar-content .btn-success:hover {
    background-color: #40c057 !important;
    border-color: #40c057 !important;
    color: #ffffff !important;
}
html[data-bs-theme="dark"] .btn-success:focus,
html[data-bs-theme="dark"] .btn-success:focus-visible {
    box-shadow: 0 0 0 .2rem rgba(64,192,87,.25) !important;
}

html[data-bs-theme="dark"] .btn-danger,
html[data-bs-theme="dark"] aside.sidebar .btn-danger,
html[data-bs-theme="dark"] .sidebar-content .btn-danger {
    background-color: #c92a2a !important;
    border-color: #c92a2a !important;
    color: #ffffff !important;
}
html[data-bs-theme="dark"] .btn-danger:hover,
html[data-bs-theme="dark"] aside.sidebar .btn-danger:hover,
html[data-bs-theme="dark"] .sidebar-content .btn-danger:hover {
    background-color: #e03131 !important;
    border-color: #e03131 !important;
    color: #ffffff !important;
}
html[data-bs-theme="dark"] .btn-danger:focus,
html[data-bs-theme="dark"] .btn-danger:focus-visible {
    box-shadow: 0 0 0 .2rem rgba(224,49,49,.25) !important;
}

/* ---- File-input progress bar ---- */
html[data-bs-theme="dark"] .shiny-file-input-progress .progress-bar,
html[data-bs-theme="dark"] .progress-bar {
    background-color: #2f9e44 !important;
    color: #ffffff !important;
}
html[data-bs-theme="dark"] .progress {
    background-color: #2a2a2a !important;
}

/* ---- Tables ---- */
html[data-bs-theme="dark"] table,
html[data-bs-theme="dark"] .dataframe {
    color: #dcddde !important;
    background-color: #262626 !important;
}
html[data-bs-theme="dark"] th,
html[data-bs-theme="dark"] thead th {
    background-color: #2a2a2a !important;
    color: #a78bfa !important;
    border-color: #3a3a3a !important;
}
html[data-bs-theme="dark"] td {
    border-color: #3a3a3a !important;
    color: #dcddde !important;
}
html[data-bs-theme="dark"] tbody tr:hover td {
    background-color: #2f2f2f !important;
}

/* ---- Popovers / modals ---- */
html[data-bs-theme="dark"] .popover,
html[data-bs-theme="dark"] .tooltip-inner,
html[data-bs-theme="dark"] .modal-content {
    background-color: #262626 !important;
    border-color: #3a3a3a !important;
    color: #dcddde !important;
}
html[data-bs-theme="dark"] .popover-header,
html[data-bs-theme="dark"] .modal-header {
    background-color: #2a2a2a !important;
    border-bottom-color: #3a3a3a !important;
    color: #ede9fe !important;
}
html[data-bs-theme="dark"] .modal-footer {
    background-color: #2a2a2a !important;
    border-top-color: #3a3a3a !important;
}
html[data-bs-theme="dark"] .popover-body,
html[data-bs-theme="dark"] .modal-body {
    background-color: #262626 !important;
    color: #dcddde !important;
}

/* ---- Progress ---- */
html[data-bs-theme="dark"] .progress {
    background-color: #2a2a2a !important;
}
html[data-bs-theme="dark"] .progress-bar {
    background-color: #7c3aed !important;
}

/* ---- Headings ---- */
html[data-bs-theme="dark"] h1,
html[data-bs-theme="dark"] h2,
html[data-bs-theme="dark"] h3,
html[data-bs-theme="dark"] h4,
html[data-bs-theme="dark"] h5,
html[data-bs-theme="dark"] h6 {
    color: #ede9fe !important;
}

/* ---- Switch toggle (purple accent) ---- */
html[data-bs-theme="dark"] .form-switch .form-check-input:checked {
    background-color: #7c3aed !important;
    border-color: #7c3aed !important;
}
html[data-bs-theme="dark"] .form-check-input:checked {
    background-color: #7c3aed !important;
    border-color: #7c3aed !important;
}

/* ---- Dropdown menus ---- */
html[data-bs-theme="dark"] .dropdown-menu {
    background-color: #262626 !important;
    border-color: #3a3a3a !important;
    color: #dcddde !important;
}
html[data-bs-theme="dark"] .dropdown-item {
    color: #dcddde !important;
}
html[data-bs-theme="dark"] .dropdown-item:hover,
html[data-bs-theme="dark"] .dropdown-item:focus {
    background-color: #3a3a3a !important;
    color: #ede9fe !important;
}

/* ---- Alerts ---- */
html[data-bs-theme="dark"] .alert {
    background-color: #2a2a2a !important;
    border-color: #3a3a3a !important;
    color: #dcddde !important;
}

/* ---- Horizontal rules / borders ---- */
html[data-bs-theme="dark"] hr {
    border-color: #3a3a3a !important;
}
"""

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_dark_mode(id="dark_mode"),
        ui.input_action_button("language_toggle", "English", icon=icon_svg("globe")),
        ui.output_ui("loc_dynamic_model_select"),
        ui.output_ui("loc_llm_switch"),
        ui.output_ui("loc_analyse_speakers_switches"),
        ui.output_ui("loc_display_cost_prediction"),
        ui.output_ui("loc_button_analysis"),
        ui.output_text("start_analysis"),
        ui.output_ui("show_report_download_button"),
        ui.output_ui("loc_button_import_session"),
        ui.output_ui("loc_button_export_session"),
        ui.output_ui("loc_button_reset"),
        #title="Controls",
    ),
    ui.head_content(
        # Leere Inline-Favicon, damit der Browser keinen 404er-Request
        # nach /favicon.ico mehr sendet.
        ui.tags.link(rel="icon", href="data:,"),
        ui.tags.style(OBSIDIAN_CSS),
        ui.tags.script("""
(function () {
  var DARK_BG = '#1e1e1e';
  var DARK_FG = '#dcddde';
  var LIGHT_BG = '#f8f8f800';

  // --- Rewrite bslib's style.css rule in place -------------------------
  // The inline-!important strategy below should win the cascade, but some
  // browsers still display the style.css source rule in DevTools. Walking
  // the CSSOM and patching the --bslib-sidebar-main-bg value directly
  // makes the change visible at the source as well.
  function patchBslibStylesheet() {
    for (var i = 0; i < document.styleSheets.length; i++) {
      var sheet = document.styleSheets[i];
      var rules;
      try { rules = sheet.cssRules || sheet.rules; } catch (e) { continue; }
      if (!rules) continue;
      for (var j = 0; j < rules.length; j++) {
        var rule = rules[j];
        if (!rule || !rule.style) continue;
        try {
          if (rule.style.getPropertyValue('--bslib-sidebar-main-bg')) {
            rule.style.setProperty('--bslib-sidebar-main-bg', LIGHT_BG, 'important');
          }
        } catch (e) { /* cross-origin or read-only */ }
      }
    }
  }
  patchBslibStylesheet();
  [50, 200, 600, 1500, 3000].forEach(function (ms) { setTimeout(patchBslibStylesheet, ms); });

  // --- Append an override <style> at the end of <head> so it wins source
  // order against bslib's bundled stylesheet.
  var override = document.createElement('style');
  override.setAttribute('data-tt-override', 'bslib-sidebar-main-bg');
  override.textContent =
    ':root, .bslib-sidebar-layout, html .bslib-sidebar-layout, html body .bslib-sidebar-layout {' +
    '  --bslib-sidebar-main-bg: ' + LIGHT_BG + ' !important;' +
    '}';
  (document.head || document.documentElement).appendChild(override);

  var SELECTORS = [
    'html',
    'body',
    'main.bslib-page-main',
    'div.main',
    '.bslib-sidebar-layout',
    '.bslib-sidebar-layout > .main',
    '.bslib-page-fill',
    '.bslib-page-sidebar',
    '.tab-content',
    '.tab-pane.active'
  ];

  function applyTheme() {
    var isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
    SELECTORS.forEach(function (sel) {
      try {
        document.querySelectorAll(sel).forEach(function (el) {
          el.style.setProperty('background-color', isDark ? DARK_BG : '', 'important');
          el.style.setProperty('color', isDark ? DARK_FG : '', 'important');
        });
      } catch (e) { /* ignore bad selectors */ }
    });
    // bslib reads --_main-bg / --bslib-sidebar-main-bg off .bslib-sidebar-layout
    // to colour the .main container. Force them inline so nothing can override.
    // In light mode, use #f8f8f800 (fully transparent) instead of clearing —
    // clearing falls back to bslib's style.css default of #f8f8f8 (opaque).
    var LIGHT_TRANSPARENT = '#f8f8f800';
    document.querySelectorAll('.bslib-sidebar-layout').forEach(function (el) {
      el.style.setProperty('--_main-bg', isDark ? DARK_BG : LIGHT_TRANSPARENT, 'important');
      el.style.setProperty('--bslib-sidebar-main-bg', isDark ? DARK_BG : LIGHT_TRANSPARENT, 'important');
      el.style.setProperty('--_main-fg', isDark ? DARK_FG : '', 'important');
    });
  }

  new MutationObserver(applyTheme).observe(
    document.documentElement,
    { attributes: true, attributeFilter: ['data-bs-theme'] }
  );
  // Run immediately, plus on DOM ready, plus on a few post-load ticks to
  // catch async-injected bslib containers.
  applyTheme();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyTheme);
  }
  window.addEventListener('load', applyTheme);
  [50, 200, 600, 1500, 3000].forEach(function (ms) { setTimeout(applyTheme, ms); });
})();
""")
    ),

    # Main Content Area with Tabs for Analysis, Results, and Options
    ui.navset_tab(  
        ui.nav_panel(ui.output_text("loc_title_analysis"),
            ui.card(     
            ui.layout_columns(
                # Group and Transcript Metadata
                ui.card(
                ui.card_header(ui.output_ui("loc_general_info")),
                ui.output_ui("loc_group_id"),
                ui.output_ui("loc_num_pupils"),
                ui.output_ui("loc_name_teacher"),
                ),
                # Document Upload for Transcript and Codebook
                ui.card(
                    ui.card_header(ui.output_ui("loc_document_input")),
                    ui.output_ui("loc_upload_transcript"),
                    ui.output_ui("loc_upload_codebook"),
                ),
            ),     
            ),
            # Preview of Codebook and Transcript
            ui.card(
                ui.card_header(ui.output_ui("loc_preview_codebook")),
                ui.output_ui("show_codebook_preview"),   
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.output_ui("loc_general_transcript")),
                ui.output_ui("show_transcript_preview"),   
                full_screen=True,
            ),
            icon=icon_svg("brain")
        ),
        # Results Tab with Quantitative and Qualitative Analysis
        ui.nav_panel(ui.output_text("loc_title_results"),
            ui.card(
                ui.card_header(ui.output_ui("loc_quantitative_analysis")),
                # General group satistics 
                ui.layout_column_wrap(
                    ui.output_ui("loc_group_id_display"),
                    ui.output_ui("loc_class_size"),
                    ui.output_ui("loc_num_participants"),
                    ui.output_ui("loc_participation_rate"),                   
                    fill=False,
                ),
                ),
                # Quantitative Stats for Conversation Distribution
                ui.layout_columns(
                    # Conversation Distribution Plot
                    ui.card(
                        ui.card_header(ui.output_ui("loc_distribution_of_turns")),
                        ui.output_plot("sim_stats_plot"),
                        full_screen=True,
                    ),
                    # Conversation Statistics for Teacher and Pupils
                    ui.card(
                        ui.card_header(ui.output_ui("loc_interaction_turns")),
                        ui.layout_column_wrap(
                            ui.card(
                            ui.card_header(ui.output_ui("loc_teacher")),
                            ui.output_ui("loc_teacher_turns"),
                            ui.output_text("teacher_turns"),
                            ui.output_ui("loc_teacher_turns_length"),
                            ui.output_text("teacher_turns_length"),

                            ),
                            ui.card(
                            ui.card_header(ui.output_ui("loc_pupils")),
                            ui.output_ui("loc_pupils_turns"),
                            ui.output_text("pupils_turns"),
                            ui.output_ui("loc_pupils_turns_length"),
                            ui.output_text("pupils_turns_length"),
                            ),
                        ),
                        full_screen=True,
                    ),
                    col_widths=[4, 8]
                ),
            # Qualitative Analysis of Teacher's Impulses
            # Quick Stats for Teacher's Impulses
            ui.card(
                ui.card_header(ui.output_ui("loc_qualitative_analysis")),
                ui.layout_columns(
                    ui.value_box(
                        ui.output_ui("loc_impulses_count"),
                        ui.output_text("teacher_impulses"),
                        showcase=icon_svg("square-poll-vertical")
                    ),
                    ui.value_box(
                        ui.output_ui("loc_coded_impulses"),
                        ui.output_text("teacher_impulses_coded"),
                        showcase=icon_svg("hashtag")
                    ),
                    ui.value_box(
                        ui.output_ui("loc_most_frequent_codes"),
                        ui.output_text("code_most_used"),
                        showcase=icon_svg("ranking-star")
                    ),
                    ui.value_box(
                        ui.output_ui("loc_teacher_talking_rate"),
                        ui.output_ui("teacher_share_ui"),
                        showcase=icon_svg("user-tie")
                    ),
                    col_widths=[3]
                ),   
                ui.layout_columns(
                    # Qualitative Statistics Plot for Coded Impulses
                    ui.card(
                        ui.card_header(ui.output_ui("loc_impulses_distribution")),
                        ui.row(
                            ui.output_plot("qualitative_stats_plot"),
                        ),
                        # Explanation of Codes
                        ui.row(
                            ui.output_ui("code_legend"),
                        ),
                        full_screen=True,
                    ),
                    # DataFrame of Coded Impulses
                    ui.card(
                        ui.card_header(ui.output_ui("loc_impulses_coding")),
                        ui.output_ui("quali_stats_df"),
                        full_screen=True,
                    ),
                ),          
            ),
            icon=icon_svg("chart-bar")
        ),
        # Testing Tab: intercoder agreement (Cohen's kappa) between two reports
        ui.nav_panel(ui.output_text("loc_title_testing"),
            ui.card(
                ui.card_header(ui.output_ui("loc_testing_header")),
                ui.output_ui("loc_testing_intro"),
                ui.layout_columns(
                    ui.output_ui("loc_upload_report_a"),
                    ui.output_ui("loc_upload_report_b"),
                ),
                ui.output_ui("testing_summary"),
            ),
            ui.card(
                ui.card_header(ui.output_ui("loc_testing_kappa")),
                ui.output_ui("testing_kappa_value"),
            ),
            ui.card(
                ui.card_header(ui.output_ui("loc_testing_confusion")),
                ui.output_ui("testing_confusion_table"),
                full_screen=True,
            ),
            icon=icon_svg("scale-balanced"),
        ),
        # Options Tab for API Configuration and Custom Prompts
        ui.nav_panel(ui.output_text("loc_title_options"),
            # API Configuration 
            ui.card(
                ui.card_header(ui.output_ui("loc_api_configuration")),
                ui.layout_columns(
                    # Select between OpenAI and Groq API    
                    ui.card(
                        ui.output_text("loc_api_select_title"),
                        ui.output_ui("loc_api_select")
                    ),
                    # Api Key Management
                    ui.card(
                        ui.output_text("loc_api_key_exists"),
                        ui.layout_columns( 
                        ui.output_ui("loc_button_change_api_key"),
                        ui.output_ui("loc_button_delete_api_key"),
                        ),
                    ),
                ),
            ),
            # Modelle für LLM-Auswahl verwalten
            ui.card(
                ui.card_header(ui.output_ui("loc_llm_models")),
                ui.output_ui("loc_load_models"),
                ui.layout_columns(
                    ui.output_ui("loc_button_add_model"),
                    ui.output_ui("loc_button_remove_model"),
                    ui.output_ui("loc_button_reset_model_selection"),
                    col_widths=[3,3]
                )

            ),
                # Prompt Management for System and User Prompt
            ui.card(
                ui.card_header(ui.output_ui("loc_custom_prompts")),
                "System-Prompt",
                ui.output_text_verbatim("system_prompt_output"),
                ui.layout_columns(
                    ui.output_ui("loc_button_change_system_prompt"),
                    ui.output_ui("loc_button_reset_system_prompt"),
                    col_widths=[2,2]
                ),
                "User-Prompt",
                ui.output_text_verbatim("user_prompt_output"),
                ui.layout_columns(
                    ui.output_ui("loc_button_change_user_prompt"),
                    ui.output_ui("loc_button_reset_user_prompt"),
                    col_widths=[2,2]
                ),                                            
            ),
            ui.card(
                ui.card_header(ui.output_ui("loc_additional_options")),
                ui.layout_columns(
                    ui.output_ui("loc_input_teacher_name_options"),
                    ui.output_ui("loc_input_group_id_options"),
                    ui.output_ui("loc_input_num_pupils_options"),
                    ui.output_ui("loc_button_reset_parameters"),
                    col_widths=[2,2,2,2]
                ),
            ),
            ui.card(
                ui.card_header(ui.output_ui("loc_app_info")),
                ui.output_ui("loc_app_info_text"),
            ),
            icon=icon_svg("gear"),
        ), 
        # Tab Identifier to Actively Switch Between Tabs
        id="main_tabs",  
    ),  

    # Incluse CSS Stylesheet
    ui.include_css(str(resource_path("static/styles.css"))),
    # Set the Title of the App-Window
    title="TalkTrace AI",
    fillable=True
)


def server(input, output, session):

    # Initialize Config Manager to handle config file
    config = ConfigManager() 
    # Define helper variables
    transcript_data = reactive.value(None)
    codebook_data = reactive.value(None)
    api_key_groq = reactive.value()
    api_key_openai = reactive.value()
    api_key_anthropic = reactive.value()
    api_key_ollama = reactive.value()
    ollama_status_refresh = reactive.value(0)
    current_api = reactive.value(config.get_current_api())
    num_participants = reactive.value(None)
    participation_rate = reactive.value(None)
    t_turns = reactive.value(None)
    t_turns_length = reactive.value(None)
    t_turns_length_mean_sd = reactive.value(None)
    p_turns = reactive.value(None)
    p_turns_length = reactive.value(None)
    p_turns_length_mean_sd = reactive.value(None)
    stats = reactive.value(None)
    stats_per_speaker = reactive.value(None)
    llm_analysis_data = reactive.value([])
    model = reactive.value(config.get_current_model())
    teacher_impulses_count = reactive.value(None)
    analysis_state = reactive.value(False)
    analysis_llm_state = reactive.value(False)
    sim_plot = reactive.value(None)
    qual_plot = reactive.value()
    qual_stats_df = reactive.value(None)
    placeholder_plot = reactive.value()
    model_deleted = reactive.value(0) # for reactivitiy of model selection after model deletion
    current_lang = reactive.value(config.get_localization()["current_language"])
    code_legend_storage = reactive.value("Legende nicht ausgelesen")
    estimated_cost = reactive.value(None)
    token_count = reactive.value(None)
    report_a_df = reactive.value(None)
    report_b_df = reactive.value(None)
    report_a_error = reactive.value(None)
    report_b_error = reactive.value(None)

    ### Localization
    # Helper function to get translated text
    def t(section, key):
        return TRANSLATIONS[current_lang.get()][section][key] 

    # Update language based on user selection
    @reactive.effect
    @reactive.event(input.language_toggle)
    def _():
        req(input.language_toggle())
        # Toggle between languages
        new_lang = "en" if current_lang.get() == "de" else "de"
        # Update button text and icon
        if new_lang == "en":
            ui.update_action_button(
                "language_toggle",
                label="Deutsch",
                icon=icon_svg("globe")
            )
            config.set_localization('current_language', 'en')
        else:
            ui.update_action_button(
                "language_toggle",
                label="English",
                icon=icon_svg("globe")
            )
            config.set_localization('current_language', 'de')

        current_lang.set("de" if new_lang == "de" else "en")

        # Prompt-Basis an die neue Sprache anpassen
        prompts_new = config.get_prompts(language=current_lang.get())
        system_prompt.set(prompts_new['system'])
        user_prompt.set(prompts_new['user'])


    # Define Baseline System Prompt
    system_prompt = reactive.value(config.get_prompts()['system'])

    # Define Baseline User Prompt
    user_prompt = reactive.value(config.get_prompts()['user'])

    # Fill Fields from Config File
    ui.update_text("name_group", value=config.get_parameters()['group_id'])
    ui.update_numeric("num_pupils", value=config.get_parameters()['num_pupils'])
    ui.update_text("name_teacher", value=config.get_parameters()['teacher_name'])
    ui.update_text("name_teacher_options", value=config.get_parameters()['teacher_name'])
    ui.update_text("name_group_options", value=config.get_parameters()['group_id'])
    ui.update_numeric("num_pupils_options", value=config.get_parameters()['num_pupils'])
    ui.update_action_button("language_toggle", icon=icon_svg("globe"), label="English" if config.get_localization()['current_language'] == 'de' else "Deutsch")


    try:
        api_key_openai.set(keyring.get_password("talktrace", "api_key_openai"))
    except keyring.errors.PasswordDeleteError:
        pass

    try:
        api_key_groq.set(keyring.get_password("talktrace", "api_key_groq"))
    except keyring.errors.PasswordDeleteError:
        pass

    try:
        api_key_anthropic.set(keyring.get_password("talktrace", "api_key_anthropic"))
    except keyring.errors.PasswordDeleteError:
        pass

    try:
        api_key_ollama.set(keyring.get_password("talktrace", "api_key_ollama"))
    except keyring.errors.PasswordDeleteError:
        pass


    ### Sidebar --------------------------------------------------------
    # Model Selection
    @render.ui
    def loc_dynamic_model_select():
        return ui.input_select("model_select", t("sidebar", "model_select"), choices=select_api_choices(), selected=config.get_current_model())

    
    @reactive.effect()
    def update_current_model():
        model.set(input.model_select())
        config.set_current_model(input.model_select())
    

    # LLM Analyse
    @render.ui
    def loc_llm_switch():
        return ui.input_switch("llm_switch", t("sidebar", "llm_switch"), True)


    # Sprechakt-Auswahl: nur sichtbar, wenn LLM-Analyse aktiv ist
    @render.ui
    def loc_analyse_speakers_switches():
        if not input.llm_switch():
            return None
        return ui.TagList(
            ui.input_switch("analyse_teacher_switch", t("sidebar", "analyse_teacher_switch"), True),
            ui.input_switch("analyse_students_switch", t("sidebar", "analyse_students_switch"), True),
        )


    # Effektive Prompts: Basis-Prompt + Zusatzanweisung je nach Sprecher-Auswahl.
    # Wird sowohl in der Options-Anzeige als auch beim LLM-Call verwendet,
    # damit der User sieht, was tatsächlich ans Modell geschickt wird.
    def _speaker_flags():
        # Switches werden nur gerendert, wenn llm_switch aktiv ist;
        # fallback auf True (Default), solange sie nicht existieren.
        try:
            teacher = bool(input.analyse_teacher_switch())
        except Exception:
            teacher = True
        try:
            students = bool(input.analyse_students_switch())
        except Exception:
            students = True
        return teacher, students

    def _speaker_filter_suffix(kind: str = "system"):
        teacher, students = _speaker_flags()
        prefix = "user_prompt_filter" if kind == "user" else "prompt_filter"
        if teacher and students:
            return ""
        if teacher and not students:
            return t("sidebar", f"{prefix}_teacher_only")
        if students and not teacher:
            return t("sidebar", f"{prefix}_students_only")
        return t("sidebar", f"{prefix}_none")

    @reactive.calc
    def effective_system_prompt():
        return system_prompt.get() + _speaker_filter_suffix("system")

    @reactive.calc
    def effective_user_prompt():
        return user_prompt.get() + _speaker_filter_suffix("user")


    def calculate_input_tokens(transcript, codebook, system_prompt_text, user_prompt_text):
        """Calculate approximate token count for LLM request"""
        try:
            # Use the encoding for the selected model
            if config.get_current_api() == "openai":
                try:
                    encoding = tiktoken.encoding_for_model(model.get())
                except:
                    encoding = tiktoken.get_encoding("cl100k_base")
            else:  # groq, anthropic, ollama
                encoding = tiktoken.get_encoding("cl100k_base")
            
            # Combine all text
            all_text = f"{system_prompt_text}\n{user_prompt_text}\n{str(transcript)}\n{str(codebook)}"
            
            # Count tokens
            tokens = len(encoding.encode(all_text))
            return tokens
        except Exception as e:
            print(f"Token calculation error: {e}")
            return 0


    def calculate_estimated_cost(tokens):
        """Calculate estimated cost based on token count and selected API/model"""
        pricing = config.get_api_pricing()  # Add this to ConfigManager
        api = config.get_current_api()
        current_model = model.get()
        
        if api in pricing and current_model in pricing[api]:
            rate_in = pricing[api][current_model]["input"]  # Cost per 1K tokens
            rate_out = pricing[api][current_model]["output"]
            cost = (tokens / 1000000) * rate_in + (tokens / 1000000) * rate_out * 4
            return cost
        return None


    # Update cost prediction when transcript/codebook changes
    @reactive.effect
    def update_cost_prediction():
        req(transcript_data.get() != None, codebook_data.get() != None, input.llm_switch())
        tokens = calculate_input_tokens(
            transcript_data.get(),
            codebook_data.get() or "",
            effective_system_prompt(),
            effective_user_prompt()
        )
        token_count.set(tokens)
        cost = calculate_estimated_cost(tokens)
        estimated_cost.set(cost)


    @render.text
    def loc_display_cost_prediction():
        req(transcript_data.get() != None, codebook_data.get() != None)
        if input.llm_switch():
            tokens = token_count.get()
            cost = estimated_cost.get()
            if tokens and cost:
                return f"{t("sidebar", "tokens_aprox")} {tokens:} {t("sidebar", "cost_prediction")}: {cost:.4f} €"
        return ""


    # Start Analysis Button
    @render.ui
    def loc_button_analysis():
        return ui.input_action_button("button_analysis", t("sidebar", "button_analysis"), icon=icon_svg("magnifying-glass-chart"), class_="btn-success")

    # Shared analysis function
    async def run_analysis():
        req(transcript_data.get() != None)
        # Progress bar to indicate the analysis steps
        with ui.Progress(min=1, max=4) as p:
            p.set(message=t("system_prompts", "analysis_running"), detail=t("system_prompts", "wait"))

            transcript = transcript_data.get()
            teacher_name = input.name_teacher()

            # Quantitative Stats in Threads rechnen, damit der Event-Loop frei
            # bleibt und sie ggf. parallel zum LLM-Call laufen können.
            def _compute_stats():
                return {
                    "num_participants": count_pupils(transcript),
                    "stats": dialog_stats(transcript, teacher_name),
                    "stats_per_speaker": dialog_stats_per_speaker(transcript, teacher_name),
                }

            stats_task = asyncio.create_task(asyncio.to_thread(_compute_stats))

            # LLM-Call parallel starten, damit Statistik-Berechnung und API-Call
            # gleichzeitig laufen. `to_thread` verhindert, dass der synchrone
            # Provider-SDK-Call den Shiny-Event-Loop blockiert.
            llm_task = None
            if input.llm_switch():
                req(input.codebook())
                teacher_on, students_on = _speaker_flags()
                req(teacher_on or students_on)
                sys_p = effective_system_prompt()
                usr_p = effective_user_prompt()
                current_api = config.get_current_api()
                cb = codebook_data.get()
                mdl = model.get()

                # Progress-Callback: wird aus Worker-Thread heraus aufgerufen.
                # Shiny-Progress-Updates aus Threads sind nicht thread-safe,
                # daher nur als Debug-Zähler – Main-Thread aktualisiert Progress
                # vor/nach dem Call. Haken bewusst leichtgewichtig.
                stream_chunks = {"n": 0}
                def _stream_progress(n):
                    stream_chunks["n"] = n

                if current_api == "groq":
                    req(api_key_groq.get() != None)
                    client = get_groq_client(api_key_groq.get())
                    llm_task = asyncio.create_task(asyncio.to_thread(
                        llm_analysis_groq, sys_p, usr_p, mdl, transcript, cb, client))
                elif current_api == "openai":
                    req(api_key_openai.get() != None)
                    client = get_openai_client(api_key_openai.get())
                    llm_task = asyncio.create_task(asyncio.to_thread(
                        llm_analysis_openai, sys_p, usr_p, mdl, transcript, cb, client))
                elif current_api == "anthropic":
                    req(api_key_anthropic.get() != None)
                    client = get_anthropic_client(api_key_anthropic.get())
                    llm_task = asyncio.create_task(asyncio.to_thread(
                        llm_analysis_anthropic, sys_p, usr_p, mdl, transcript, cb, client, _stream_progress))
                elif current_api == "ollama":
                    llm_task = asyncio.create_task(asyncio.to_thread(
                        llm_analysis_ollama, sys_p, usr_p, mdl, transcript, cb))

            # Zuerst Statistik einsammeln (läuft parallel zum LLM-Call).
            stats_result = await stats_task
            num_participants.set(stats_result["num_participants"])
            stats.set(stats_result["stats"])
            stats_per_speaker.set(stats_result["stats_per_speaker"])
            teacher_impulses_count.set(count_teacher_impulses(stats.get(), teacher_name))
            p.set(1, message=t("system_prompts", "calculating"))

            # Participation rate + per-speaker turn stats sofort berechnen,
            # damit sie für Report-Download und Session-Export verfügbar sind,
            # auch ohne dass der Results-Tab gerendert wurde.
            num_p = num_participants.get() or 0
            num_class = input.num_pupils() or 0
            participation_rate.set((num_p / num_class * 100) if num_class else 0)

            df_stats = stats.get()

            def _safe(speaker, col, default=0):
                m = df_stats.loc[df_stats['Sprecher'] == speaker, col]
                return m.values[0] if not m.empty else default

            t_turns.set(_safe(teacher_name, 'Anzahl_Beitraege'))
            t_turns_length.set(round(_safe(teacher_name, 'Durchschnitt_Woerter'), 1))
            t_turns_length_mean_sd.set(round(_safe(teacher_name, 'Median_Woerter'), 1))
            p_turns.set(_safe("Schüler:innen", 'Anzahl_Beitraege'))
            p_turns_length.set(round(_safe("Schüler:innen", 'Durchschnitt_Woerter'), 1))
            p_turns_length_mean_sd.set(_safe("Schüler:innen", 'Median_Woerter'))

            p.set(2, message=t("system_prompts", "waiting_LLM"))

            # Auf LLM-Resultat warten, falls aktiviert.
            if llm_task is not None:
                llm_response = await llm_task

                if llm_response is None:
                    llm_response = json.dumps({"error": "No API provider matched or no response received."})

                if '"error":' in llm_response:
                    return f"{t("system_prompts", "error")}: {json.loads(llm_response)['error']}. {t("system_prompts", "try_again")}"

                existing_data = llm_analysis_data.get()
                new_data = json.loads(llm_response)
                # Handle responses that are a bare list instead of {"analysis": [...]}
                if isinstance(new_data, list):
                    new_data = {"analysis": new_data}
                # Ensure the "analysis" key exists; an empty list is valid
                # (e.g. transcripts with no codable turns).
                analysis_items = new_data.get("analysis", []) if isinstance(new_data, dict) else []
                if not isinstance(analysis_items, list):
                    analysis_items = []
                print(f"[LLM ANALYSIS] provider={config.get_current_api()} model={model.get()} returned {len(analysis_items)} coded items")
                if len(analysis_items) == 0:
                    # Surface to the UI so the user knows the model returned an empty coding.
                    return f"{t('system_prompts', 'error')}: LLM returned 0 coded items. {t('system_prompts', 'try_again')}"
                # Back-fill Sprecher if an older model returned only 3 fields.
                for item in analysis_items:
                    if isinstance(item, dict) and "Sprecher" not in item:
                        item["Sprecher"] = ""
                new_data_df = pd.DataFrame(analysis_items, columns=['#', "Sprecher", "Shortcode", "Impuls"])

                existing_data.append(new_data_df)
                llm_analysis_data.set(list(existing_data)) # Important to Set as a List to Avoid Reactivity Issues, Due to Immutability Logic of Python!!!
                analysis_llm_state.set(True)
            p.set(4, message=t("sidebar", "analysis_completed"))
            # Mark Analysis as Completed
            analysis_state.set(True)
        # Automatically Switch to Results Tab
        ui.update_navs("main_tabs", selected='<div id="loc_title_results" class="shiny-text-output"></div>')
        return t("sidebar", "analysis_completed")

    # Analyse starten
    @render.text
    @reactive.event(input.button_analysis)
    async def start_analysis():
        return await run_analysis()

    @render.ui
    def show_report_download_button():
        req(analysis_state.get())
        return ui.download_button("download_report", t("sidebar", "download_report"), icon = icon_svg("download")),


    @render.download(filename=lambda: f"{date.today().isoformat()} - TalkTrace AI {t("results", "results_group")} {input.name_group()}.docx")
    def download_report():
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        tmp_file.close()
        if llm_analysis_data.get():
            generate_report2(tmp_file.name, input.name_group(), input.num_pupils(), num_participants.get(), participation_rate.get(), {"num": t_turns.get(), "words": t_turns_length.get(), "mean_sd": t_turns_length_mean_sd.get()}, {"num": p_turns.get(), "words": p_turns_length.get(), "mean_sd": p_turns_length_mean_sd.get()}, sim_plot.get(), teacher_impulses_count.get(), code_legend_storage.get(), True, qual_plot.get(), qual_stats_df.get(), model_name=model.get() or "")

        else:
            generate_report2(tmp_file.name, input.name_group(), input.num_pupils(), num_participants.get(), participation_rate.get(), {"num": t_turns.get(), "words": t_turns_length.get(), "mean_sd": t_turns_length_mean_sd.get()}, {"num": p_turns.get(), "words": p_turns_length.get(), "mean_sd": p_turns_length_mean_sd.get()}, sim_plot.get(), teacher_impulses_count.get(), llm_analysis=False, caption=code_legend_storage.get())

        return tmp_file.name


    # Import Session
    @render.ui
    def loc_button_import_session():
        return ui.input_file("button_import_session", t("sidebar", "import_session"), accept=[".pkl"], multiple=False, placeholder=t("analysis", "placeholder"), button_label=t("analysis", "browse")),


    @reactive.effect
  #  @reactive.event(input.button_import_session)
    async def button_import_session():
        
        file = input.button_import_session()

        if not file:
            return
    
        with open(file[0]["datapath"], "rb") as f:
            session_data = pickle.load(f)
        
        # Set the reactive values
        with reactive.isolate(): 
            try:
                transcript_data.set(session_data.get("transcript_data"))
                num_participants.set(session_data.get("num_participants"))
                participation_rate.set(session_data.get("participation_rate"))
                stats.set(session_data.get("stats"))
                llm_analysis_data.set(session_data.get("llm_analysis_data"))
                analysis_llm_state.set(session_data.get("analysis_llm_state"))
                placeholder_plot.set(session_data.get("placeholder_plot"))
                code_legend_storage.set(session_data.get("code_legend_storage"))
                ui.update_switch("llm_switch", value=False)
            except Exception as e:
                pass

        await run_analysis()
        '''
        m = ui.modal(  
                t("analysis", "modal_restart_analysis"),  
                title=t("analysis", "modal_title_attention"), 
                easy_close=True,
                footer=ui.modal_button("OK", class_="btn-success")  
            )  
        ui.modal_show(m)  
        '''

    # Export Session
    @render.ui
    def loc_button_export_session():
        return ui.download_button("button_export_session", t("sidebar", "export_session"), icon = icon_svg("file-export")),

    
    @render.download(filename=lambda: f"{date.today().isoformat()} - TalkTrace AI Session - {t("results", "results_group")} {input.name_group()} - {config.get_current_model}.pkl")
    def button_export_session():
        session_data = {
            "transcript_data": transcript_data.get(),
            "num_participants": num_participants.get(),
            "participation_rate": participation_rate.get(),
            "stats": stats.get(),
            "llm_analysis_data": llm_analysis_data.get(),
            "analysis_llm_state": analysis_llm_state.get(),
            "code_legend_storage": code_legend_storage.get(),
        }
        
        # serialize the dictionary to a pickle file
        with open("session_dump.pkl", "wb") as f:
            pickle.dump(session_data, f)
        return "session_dump.pkl"


    # Reset Session
    @render.ui
    def loc_button_reset():
        return ui.input_action_button("button_reset", t("sidebar", "reset_session"), icon = icon_svg("arrow-rotate-left"), class_="btn-danger"),

    @reactive.effect
    @reactive.event(input.button_reset)
    def reset_session():
        m = ui.modal(
            t("analysis", "modal_reset_session"),
            title=t("analysis", "modal_title_reset"),
            easy_close=True,
            footer=(ui.input_action_button("button_confirm_session_reset", t("analysis", "modal_confirm_reset"), class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"), class_="btn-danger")),
        )
        ui.modal_show(m)

    # Reset all reactive values to their initial state
    @reactive.effect
    @reactive.event(input.button_confirm_session_reset)
    def confirm_reset_session():
        transcript_data.set(None)
        codebook_data.set(None)
        num_participants.set(None)
        participation_rate.set(None)
        stats.set(None)
        stats_per_speaker.set(None)
        llm_analysis_data.set([])
        teacher_impulses_count.set(None)
        analysis_state.set(False)
        analysis_llm_state.set(False)
        sim_plot.set(None)
        qual_plot.set(None)
        qual_stats_df.set(None)
        placeholder_plot.set(None)
        code_legend_storage.set("Legende nicht ausgelesen")
        ui.update_text("name_group", value=config.get_parameters()['group_id'])
        ui.update_numeric("num_pupils", value=config.get_parameters()['num_pupils'])
        ui.update_text("name_teacher", value=config.get_parameters()['teacher_name'])

        # Close modal and go back to Analysis Pane
        ui.modal_remove()
        ui.update_navs("main_tabs", selected='<div id="loc_title_analysis" class="shiny-text-output"></div>')

    ### Analyse --------------------------------------------------------

    @render.text
    def loc_title_analysis():
        return (t("analysis", "tab_title"))

    # Allgemeine Informationen
    @render.ui
    def loc_general_info():
        return ui.p(t("analysis", "general_info"))
    
    @render.ui
    def loc_group_id():
        return ui.input_text("name_group", t("analysis", "group_id"), "B1")
    
    @render.ui
    def loc_num_pupils():
        return ui.input_numeric("num_pupils", t("analysis", "num_pupils"), 25, min=1, max=100)

    @render.ui
    def loc_name_teacher():
        return ui.input_text("name_teacher", t("analysis", "name_teacher"), config.get_parameters()['teacher_name'])
    
    # Dokumenteneingabe
    @render.ui
    def loc_document_input():
        return ui.p(t("analysis", "document_input"))

    # Transkript Upload
    @render.ui
    def loc_upload_transcript():
        return ui.input_file(
            "transcript",
            t("analysis", "upload_transcript"),
            multiple=False,
            accept=[".txt", ".docx", ".pdf"],
            button_label=t("analysis", "browse"),
            placeholder=t("analysis", "placeholder"),
        )

    # Transkript verarbeiten
    @reactive.effect
    @reactive.event(input.transcript)
    def process_transcript():
        file = input.transcript()
        if file is not None:
            transcript_data.set(import_file(file[0]))
    

    # Warnung bei fehlendem Transkript   
    @reactive.effect
    @reactive.event(input.button_analysis)
    def _():
        if transcript_data.get() == None:
            m = ui.modal(  
                t("analysis", "modal_upload_transcript_first"),  
                title=t("analysis", "modal_title_error"), 
                easy_close=True,
                footer=ui.modal_button("OK",  class_="btn-success"),  
            )  
            ui.modal_show(m)  


    # Codebuch Upload
    @render.ui
    def loc_upload_codebook():
        return ui.input_file(
            "codebook",
            t("analysis", "upload_codebook"),
            multiple=False,
            accept=[".txt", ".docx", ".pdf"],
            button_label=t("analysis", "browse"),
            placeholder=t("analysis", "placeholder"),
        )
    

    # Codebuch verarbeiten
    @reactive.effect
    @reactive.event(input.codebook)
    def process_codebook():
        file = input.codebook()
        if file is not None:
            codebook_data.set(import_file(file[0]))


    # Warnung bei fehlendem Codebuch
    @reactive.effect
    @reactive.event(input.button_analysis)
    def _():
        req(input.llm_switch())
        if codebook_data.get() == None:
            m = ui.modal(  
                t("analysis", "modal_upload_codebook_first"),  
                title=t("analysis", "modal_title_error"),  
                easy_close=True,
                footer=ui.modal_button("OK",  class_="btn-success"),  
            )  
            ui.modal_show(m)  


    # Vorschau Codebuch
    @render.ui
    def loc_preview_codebook():
        return ui.p(t("analysis", "preview_codebook"))


    @render.ui
    def show_codebook_preview():
        data = codebook_data.get()
        if data is None:
            return t("analysis", "placeholder_codebook")
        elif isinstance(data, list):
            return ui.output_table("codebook_preview")
        else:
            return ui.pre(str(data))

    @render.table
    def codebook_preview():
        req(codebook_data.get() != None)
        return pd.DataFrame(codebook_data.get())


    # Vorschau Transkript
    @render.ui
    def loc_general_transcript():
        return ui.p(t("analysis", "preview_transcript"))


    @render.ui
    def show_transcript_preview():
        if transcript_data.get() == None:
            return t("analysis", "placeholder_transcript")
        else:
            return transcript_data.get()


    ### Testen (Intercoder-Übereinstimmung) ------------------------------
    @render.text
    def loc_title_testing():
        return t("testing", "tab_title")

    @render.ui
    def loc_testing_header():
        return ui.p(t("testing", "section_header"))

    @render.ui
    def loc_testing_intro():
        return ui.p(t("testing", "intro"))

    @render.ui
    def loc_testing_kappa():
        return ui.p(t("testing", "kappa_header"))

    @render.ui
    def loc_testing_confusion():
        return ui.p(t("testing", "confusion_header"))

    @render.ui
    def loc_upload_report_a():
        return ui.input_file(
            "report_a",
            t("testing", "upload_report_a"),
            multiple=False,
            accept=[".docx"],
            button_label=t("analysis", "browse"),
            placeholder=t("testing", "placeholder_report"),
        )

    @render.ui
    def loc_upload_report_b():
        return ui.input_file(
            "report_b",
            t("testing", "upload_report_b"),
            multiple=False,
            accept=[".docx"],
            button_label=t("analysis", "browse"),
            placeholder=t("testing", "placeholder_report"),
        )

    @reactive.effect
    @reactive.event(input.report_a)
    def _process_report_a():
        f = input.report_a()
        if not f:
            return
        try:
            report_a_df.set(parse_report_impulses(f[0]['datapath']))
            report_a_error.set(None)
        except Exception:
            report_a_df.set(None)
            report_a_error.set(t("testing", "parse_error_no_table"))

    @reactive.effect
    @reactive.event(input.report_b)
    def _process_report_b():
        f = input.report_b()
        if not f:
            return
        try:
            report_b_df.set(parse_report_impulses(f[0]['datapath']))
            report_b_error.set(None)
        except Exception:
            report_b_df.set(None)
            report_b_error.set(t("testing", "parse_error_no_table"))

    @reactive.calc
    def _agreement():
        a = report_a_df.get()
        b = report_b_df.get()
        if a is None or b is None:
            return None
        return compute_intercoder_agreement(
            a, b, unmatched_label=t("testing", "unmatched_label")
        )

    @render.ui
    def testing_summary():
        err_a = report_a_error.get()
        err_b = report_b_error.get()
        items = []
        if err_a:
            items.append(ui.tags.div(f"Report A: {err_a}", class_="text-danger"))
        if err_b:
            items.append(ui.tags.div(f"Report B: {err_b}", class_="text-danger"))

        res = _agreement()
        if res is None:
            if not items:
                return ui.p(t("testing", "kappa_not_ready"))
            return ui.TagList(*items)

        items.append(
            ui.layout_columns(
                ui.value_box(t("testing", "summary_n_pairs"),
                             str(res["n_pairs"]), theme="primary"),
                ui.value_box(t("testing", "summary_n_both"),
                             str(res["n_both"]), theme="success"),
                ui.value_box(t("testing", "summary_only_a"),
                             str(res["n_only_a"]), theme="warning"),
                ui.value_box(t("testing", "summary_only_b"),
                             str(res["n_only_b"]), theme="warning"),
            )
        )
        return ui.TagList(*items)

    def _kappa_interpretation_key(k):
        if k < 0:    return "kappa_interpretation_poor"
        if k <= 0.2: return "kappa_interpretation_slight"
        if k <= 0.4: return "kappa_interpretation_fair"
        if k <= 0.6: return "kappa_interpretation_moderate"
        if k <= 0.8: return "kappa_interpretation_substantial"
        return "kappa_interpretation_almost_perfect"

    @render.ui
    def testing_kappa_value():
        res = _agreement()
        if res is None:
            return ui.p(t("testing", "kappa_not_ready"))
        k = res["kappa"]
        if k != k:  # NaN check
            return ui.p("κ = n/a")
        label = t("testing", _kappa_interpretation_key(k))
        return ui.TagList(
            ui.tags.div(f"κ = {k:.3f}",
                        style="font-size: 2.4rem; font-weight: 600;"),
            ui.tags.div(label, style="color: var(--bs-secondary-color);"),
        )

    @render.ui
    def testing_confusion_table():
        res = _agreement()
        if res is None:
            return ui.p(t("testing", "kappa_not_ready"))
        cm = res["confusion"]
        if cm.empty:
            return ui.p("—")
        header_cells = [ui.tags.th("A \\ B")] + [ui.tags.th(str(c)) for c in cm.columns]
        header = ui.tags.thead(ui.tags.tr(*header_cells))
        body_rows = []
        for idx, row in cm.iterrows():
            cells = [ui.tags.th(str(idx))] + [ui.tags.td(str(int(v))) for v in row.values]
            body_rows.append(ui.tags.tr(*cells))
        body = ui.tags.tbody(*body_rows)
        return ui.tags.table(header, body,
                             class_="table table-sm table-bordered table-striped")


    ### Ergebnisse --------------------------------------------------------
    # Ergebnisse Tab Titel
    @render.text
    def loc_title_results():
        return (t("results", "tab_title"))
    

    # Warnung, wenn Ergebnisse Tab ohne Analyse angeklickt wird
    @reactive.effect
    @reactive.event(input.main_tabs)
    def warn_if_results_tab_clicked():
        if input.main_tabs() == '<div id="loc_title_results" class="shiny-text-output"></div>' and not analysis_state.get():
            m = ui.modal(
                ui.p(t("results", "no_results")),
                title=t("results", "no_results_title"),
                easy_close=True,
                footer=ui.modal_button("OK",  class_="btn-success"),
                size="m"
            )
            ui.modal_show(m)
            ui.update_navs("main_tabs", selected='<div id="loc_title_analysis" class="shiny-text-output"></div>')

    # Anzeige der allgemeinen Informationen
    @render.ui
    def loc_quantitative_analysis():
        return ui.h3(t("results", "section_quantitative_analysis"))


    # Die Berechnung der Stats-Werte (t_turns, p_turns, ...) erfolgt jetzt
    # direkt in run_analysis(), damit sie auch ohne gerenderten Results-Tab
    # für Report-Download und Session-Export verfügbar sind.


    # Anzeige der Gruppen-ID
    @render.ui
    def loc_group_id_display():
        return ui.value_box(
                        ui.p(t("analysis", "group_id")),
                        ui.output_text("nameGroup"),
                        showcase=icon_svg("id-card"),
                    ),
    

    @render.text
    def nameGroup():
        return input.name_group()


    # Anzeige der Klassengröße
    @render.ui
    def loc_class_size():
        return ui.value_box(
                        ui.p(t("results", "class_size")),
                        ui.output_text("numPupils"),
                        showcase=icon_svg("user-group"),
                    ),


    @render.text
    def numPupils():
        return input.num_pupils()
    

    # Anzeige der Anzahl beteiligter Schüler:innen
    @render.ui
    def loc_num_participants():
        return ui.value_box(
                        ui.p(t("results", "num_participants")),
                        ui.output_text("numParticipants"),
                        showcase=icon_svg("user-check"),
                    ),


    @render.text
    def numParticipants():
        req(num_participants.get() != None)
        return num_participants.get()
    

    # Anzeige der Beteiligungsquote
    @render.ui
    def loc_participation_rate():
        return ui.value_box(
                        ui.p(t("results", "participation_rate")),
                        ui.output_text("participationRate"),
                        showcase=icon_svg("square-poll-vertical"),
                    ),
    
    
    @render.text
    def participationRate():
        req(participation_rate.get() is not None)
        return f"{round(participation_rate.get(), 2)} %"
    

    # Verteilung der Gesprächsbeiträge
    @render.ui
    def loc_distribution_of_turns():
        return ui.p(t("results", "distribution_of_turns"))
    

     # Create a bar plot for quantitative statistics
    @reactive.calc
    def make_sim_stats_plot():
        req(transcript_data.get() != None)

        stats_df = stats.get()
        distribution = stats_df.plot(kind='bar', x='Sprecher', y='Gesamt_Woerter', alpha=1, rot=0)
        distribution.set_xlabel(t("results", "words_total"))
        distribution.set_ylabel(t("results", "quantity"))
        distribution.set_axisbelow(True)
        distribution.grid(color='gray', axis='y')
        legend = distribution.get_legend()
        if legend is not None:
            legend.remove()
        total = stats_df['Gesamt_Woerter'].sum() or 1  # avoid div-by-zero when empty
        # Build tick labels matching whatever rows are actually present in stats_df.
        # Transcripts without a teacher have only student rows; a teacher-only
        # transcript has only the teacher row. Map by speaker name so labels
        # never mismatch the number of ticks.
        teacher_label = t("stats", "teacher")
        students_label = t("stats", "students")
        teacher_name = t("analysis", "name_teacher_var")
        tick_labels = [
            teacher_label if str(spk) == teacher_name else students_label
            for spk in stats_df['Sprecher'].tolist()
        ]
        distribution.set_xticks(range(len(tick_labels)))
        distribution.set_xticklabels(tick_labels)
        for container in distribution.containers:
            perc_labels = [f"{(bar.get_height() / total * 100):.1f}%" for bar in container]

            distribution.bar_label(container, label_type='center')
            distribution.bar_label(container, labels=perc_labels, label_type='edge')

        sim_plot.set(distribution)
        return distribution


    # Plot für Gesprächsverteilung
    @render.plot(alt="placeholder")
    def sim_stats_plot():
        if analysis_state.get() == False:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        else:
            return make_sim_stats_plot()  

    # Gesprächsstatistiken
    @render.ui
    def loc_interaction_turns():
        return ui.p(t("results", "interaction_turns"))
    
    # Lehrperson
    @render.ui
    def loc_teacher():
        return ui.p(t("results", "teacher"))


    @render.ui
    def loc_teacher_turns():
        return ui.markdown(f"**{t("results", "turn_count")}**")
    

    # Gesprächsbeiträge Lehrperson
    @render.text
    def teacher_turns():
        req(analysis_state.get(), transcript_data.get() != None)
        m = stats.get().loc[stats.get()['Sprecher'] == input.name_teacher(), 'Anzahl_Beitraege']
        return m.values[0] if not m.empty else 0
    

    # Display the TOTAL number of turns (teacher + all students).
    @render.text
    def teacher_impulses():
        req(stats.get() is not None and not stats.get().empty)
        total_turns = int(stats.get()['Anzahl_Beitraege'].sum())
        return total_turns
    

    @render.ui
    def loc_teacher_turns_length():
        return ui.markdown(f"**{t("results", "turn_length")}**")


    # Länge der Gesprächsbeiträge Lehrperson
    @render.text
    def teacher_turns_length():
        req(analysis_state.get(), transcript_data.get() != None)
        df = stats.get()
        teacher = input.name_teacher()
        avg = df.loc[df['Sprecher'] == teacher, 'Durchschnitt_Woerter']
        med = df.loc[df['Sprecher'] == teacher, 'Median_Woerter']
        return f"{round(avg.values[0], 1) if not avg.empty else 0} ({round(med.values[0], 1) if not med.empty else 0})"
    

    # Schüler:innen
    @render.ui
    def loc_pupils():
        return ui.p(t("results", "students"))
    

    @render.ui
    def loc_pupils_turns():
        return ui.markdown(f"**{t("results", "turn_count")}**")


    # Gesprächsbeiträge Schüler:innen
    @render.text
    def pupils_turns():
        req(analysis_state.get(), transcript_data.get() != None)
        m = stats.get().loc[stats.get()['Sprecher'] == "Schüler:innen", 'Anzahl_Beitraege']
        return m.values[0] if not m.empty else 0
    

    @render.ui
    def loc_pupils_turns_length():
        return ui.markdown(f"**{t("results", "turn_length")}**")
    

    # Länge der Gesprächsbeiträge Schüler:innen
    @render.text
    def pupils_turns_length():
        req(analysis_state.get(), transcript_data.get() != None)
        df = stats.get()
        avg = df.loc[df['Sprecher'] == "Schüler:innen", 'Durchschnitt_Woerter']
        med = df.loc[df['Sprecher'] == "Schüler:innen", 'Median_Woerter']
        return f"{round(avg.values[0], 1) if not avg.empty else 0} ({med.values[0] if not med.empty else 0})"
    

    # Anzeige der Qualitativen Analyse
    @render.ui
    def loc_qualitative_analysis():
        return ui.h3(t("results", "section_qualitative_analysis"))


    # Quick Stats
    @render.ui
    def loc_impulses_count():
        return ui.p(t("results", "impulses_count"))


    @render.ui
    def loc_coded_impulses():
        return ui.p(t("results", "coded_impulses"))
    

    # Display the number of impulses coded
    @render.text
    def teacher_impulses_coded():
        req(analysis_llm_state.get(), analysis_state.get())
        # Count number of rows in the dataframe
        num_impulses = qual_stats_df.get().shape[0] if qual_stats_df.get() is not None else "0"
        return num_impulses
    

    @render.ui
    def loc_most_frequent_codes():
        return ui.p(t("results", "most_frequent_codes"))
    
    # Display the most used code
    @render.text
    def code_most_used():
        req(analysis_llm_state.get(), analysis_state.get())
        # Find the most used code
        try:
            df = qual_stats_df.get()
            if df is None or df.empty:
                return t("system_prompts", "no_code")
            most_used_codes = df[t("report", "shortcode")].mode().to_list()
            return ', '.join(most_used_codes) if most_used_codes else t("system_prompts", "no_code")
        except Exception:
            return t("system_prompts", "no_code")


    @render.ui
    def loc_teacher_talking_rate():
        return ui.p(t("results", "teacher_talking_rate"))
    

    # Display the share of words spoken by teacher vs. students,
    # plus an expandable popover with a per-student breakdown.
    @render.ui
    def teacher_share_ui():
        req(stats.get() is not None and not stats.get().empty)
        df = stats.get()
        total_words = df['Gesamt_Woerter'].sum()
        teacher_name = input.name_teacher()

        tw = df.loc[df['Sprecher'] == teacher_name, 'Gesamt_Woerter']
        teacher_words = tw.values[0] if not tw.empty else 0

        sw = df.loc[df['Sprecher'] == "Schüler:innen", 'Gesamt_Woerter']
        student_words = sw.values[0] if not sw.empty else 0

        if total_words <= 0:
            return ui.span("0 %")

        t_share = round(teacher_words / total_words * 100, 1)
        s_share = round(student_words / total_words * 100, 1)

        teacher_label = t("results", "teacher")
        students_label = t("results", "students")

        # Per-student breakdown for the popover.
        per_speaker_df = stats_per_speaker.get()
        details_rows = []
        if per_speaker_df is not None and not per_speaker_df.empty:
            # Teacher row first.
            t_row = per_speaker_df.loc[per_speaker_df['Sprecher'] == teacher_name]
            if not t_row.empty:
                w = int(t_row['Gesamt_Woerter'].values[0])
                pct = round(w / total_words * 100, 1) if total_words > 0 else 0
                details_rows.append((teacher_label, w, pct))
            # Each student, sorted by speaker label (S01, S02, ...).
            student_rows = per_speaker_df.loc[per_speaker_df['Sprecher'] != teacher_name].sort_values('Sprecher')
            for _, r in student_rows.iterrows():
                w = int(r['Gesamt_Woerter'])
                pct = round(w / total_words * 100, 1) if total_words > 0 else 0
                details_rows.append((str(r['Sprecher']), w, pct))

        # Build the popover body: a compact, scrollable table.
        table_rows = [
            ui.tags.tr(
                ui.tags.th(t("results", "speaker"), style="text-align:left; padding:2px 8px;"),
                ui.tags.th(t("results", "words_total"), style="text-align:right; padding:2px 8px;"),
                ui.tags.th("%", style="text-align:right; padding:2px 8px;"),
            )
        ]
        for label, w, pct in details_rows:
            table_rows.append(
                ui.tags.tr(
                    ui.tags.td(label, style="text-align:left; padding:2px 8px;"),
                    ui.tags.td(f"{w}", style="text-align:right; padding:2px 8px;"),
                    ui.tags.td(f"{pct} %", style="text-align:right; padding:2px 8px;"),
                )
            )
        details_table = ui.tags.div(
            ui.tags.table(*table_rows, style="font-size:0.85rem; border-collapse:collapse; width:100%;"),
            style="max-height:300px; overflow-y:auto;",
        )

        summary = ui.tags.span(
            f"{teacher_label}: {t_share} % | {students_label}: {s_share} %",
            style="font-size:0.95rem;",
        )
        details_btn = ui.tags.span(
            ui.popover(
                ui.tags.a("Details ▾", href="#", style="font-size:0.8rem; margin-left:0.5rem; text-decoration:underline; cursor:pointer;"),
                details_table,
                title=t("results", "teacher_talking_rate"),
                placement="bottom",
            )
        )
        return ui.tags.div(summary, details_btn)


    # Qualitative Statistics Plot for Coded Impulses
    @render.ui
    def loc_impulses_distribution():
        ui.p(t("results", "impulses_distribution"))


    # Create a bar plot for qualitative statistics
    @reactive.calc
    def make_qualitative_stats_plot():
        req(llm_analysis_data.get())
        latest_df = llm_analysis_data.get()[-1]
        # Empty analyses (e.g. no teacher in transcript) -> placeholder figure
        if latest_df is None or latest_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            qual_plot.set(ax)
            return ax
        analysis_plot = latest_df.groupby(t("report", "shortcode")).agg(
            Anzahl=(t("report", "shortcode"), 'count'),
            ).reset_index().plot(kind='bar', x=t("report", "shortcode"), y='Anzahl', alpha=1, rot=0)
        analysis_plot.set_xlabel(t("report", "shortcode"))
        # Rotate tick labels without resetting ticks (avoids FixedLocator/labels mismatch)
        plt.setp(analysis_plot.get_xticklabels(), rotation=45, ha='right')
        analysis_plot.set_ylabel(t("report", "quantity"))
        analysis_plot.set_axisbelow(True)
        analysis_plot.grid(color='gray', axis = 'y')
        legend = analysis_plot.get_legend()
        if legend is not None:
            legend.remove()
        for container in analysis_plot.containers:
            analysis_plot.bar_label(container, label_type='edge')
        qual_plot.set(analysis_plot)
        return analysis_plot


    # Plot für qualitative Statistik
    @render.plot(alt="Noch keine Daten")
    def qualitative_stats_plot():
        if not llm_analysis_data.get():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        else:
            return make_qualitative_stats_plot()
    

    # DataFrame of Coded Impulses
    @render.ui
    def loc_impulses_coding():
        ui.p(t("results", "impulses_coding"))


    # Create a DataFrame for qualitative statistics
    @reactive.calc
    def make_qualitative_stats_df():
        req(llm_analysis_data.get())
        analysis_df = llm_analysis_data.get()[-1].copy()
        cols = ['#', t("report", "speaker"), t("report", "teacher_statement"), t("report", "shortcode")]
        # Empty analysis (no codable turns) -> return empty, properly-named df
        if analysis_df.empty:
            empty_df = pd.DataFrame(columns=cols)
            qual_stats_df.set(empty_df)
            return empty_df
        # Back-fill Sprecher column if missing (older sessions)
        if "Sprecher" not in analysis_df.columns:
            analysis_df["Sprecher"] = ""
        analysis_df['#'] = analysis_df.reset_index().index+1
        analysis_df = analysis_df[['#', "Sprecher", "Impuls", "Shortcode"]]
        analysis_df.columns = cols
        qual_stats_df.set(analysis_df)
        return analysis_df
    

# DataFrame für qualitative Statistik generieren
    @render.table()
    def qualitative_stats_df():
        return make_qualitative_stats_df()


    # DataFrame für qualitative Statistik
    @render.ui
    def quali_stats_df():
        if not llm_analysis_data.get():
            return ui.output_plot("placeholder")
        else:
            return ui.output_table("qualitative_stats_df")


    # Placeholder Plot, wenn noch keine Daten vorhanden sind
    @render.plot(alt="Noch keine Daten")
    def placeholder():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
        ax.axis('off')
        placeholder_plot.set(fig)
        return fig

    
    @render.plot(alt="Noch keine Daten")
    def placeholder2():
        return placeholder_plot.get()


    # Code-Legende aus Codebuch extrahieren
    @reactive.effect
    def extract_code_legend():
        data = codebook_data.get()
        req(data != None)
        if isinstance(data, list):
            df = pd.DataFrame(data)
            legend = [f"{code}" for code in df[df.columns[0]].unique()]
            code_legend_storage.set("; ".join(legend))
        else:
            code_legend_storage.set(str(data))
    

    # Code-Legende anzeigen
    @render.ui
    def code_legend():
        return ui.markdown(f"**{t("results", "caption")}:** {code_legend_storage.get()}")


    ### Optionen --------------------------------------------------------

    @render.text
    def loc_title_options():
        return (t("options", "tab_title"))
    

    # Api Konfiguration
    @render.ui
    def loc_api_configuration():
        return ui.p(t("options", "api_configuration"))
    

    @render.text
    def loc_api_select_title():
        return t("options", "api_select_title")


    @render.ui
    def loc_api_select():
        return ui.input_select("api_select", t("options", "api_select_title"), choices={"openai": "OpenAI", "groq": "Groq", "anthropic": "Anthropic", "ollama": "Ollama"}, selected=config.get_current_api())

    @reactive.effect
    def update_api_selection():
        config.set_current_api(input.api_select())
        current_api.set(input.api_select())

    # Anzeige, ob ein API-Key vorhanden ist
    @render.text
    def loc_api_key_exists():
        selected = input.api_select()
        if selected == "openai":
            a = api_key_openai.get()
            return t("options", "api_openai_found") if api_key_openai.get() else t("options", "api_openai_not_found")
        elif selected == "groq":
            a = api_key_groq.get()
            return t("options", "api_groq_found") if api_key_groq.get() else t("options", "api_groq_not_found")
        elif selected == "anthropic":
            a = api_key_anthropic.get()
            return t("options", "api_anthropic_found") if api_key_anthropic.get() else t("options", "api_anthropic_not_found")
        elif selected == "ollama":
            ollama_status_refresh.get()  # reactivity
            reactive.invalidate_later(600)
            url = "http://localhost:11434/"
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    if resp.status == 200:
                        return t("options", "ollama_running").format(url=url)
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            return t("options", "ollama_not_running").format(url=url)

    # API-Auswahl
    @reactive.calc
    def select_api_choices():
        deleted_model = model_deleted.get() # for reactivity/invalidation
        api_current = current_api.get() # for reactivity/invalidation
        return config.get_models(provider=config.get_current_api())
        
    # Warnung bei fehlendem API-Key
    @reactive.effect
    @reactive.event(input.button_analysis)
    def _():
        req(input.llm_switch(), input.button_analysis(), transcript_data.get() != None, codebook_data.get() != None)
        selected = input.api_select()
        missing_key = (
            (selected == "openai" and api_key_openai.get() == None) or
            (selected == "groq" and api_key_groq.get() == None) or
            (selected == "anthropic" and api_key_anthropic.get() == None)
        )
        if missing_key:
            m = ui.modal(  
                    ui.p(t("options", "no_api_key_warning")),  
                    title=t("analysis", "modal_title_error"),  
                    easy_close=True,
                    footer=ui.modal_button(t("analysis", "modal_button_close")), 
                )
            ui.modal_show(m)
            ui.update_navs("main_tabs", selected='<div id="loc_title_options" class="shiny-text-output"></div>')  

    # Button zum Ändern des API-Keys
    @render.ui
    def loc_button_change_api_key():
        if input.api_select() == "ollama":
            return ui.input_action_button("button_change_api_key", t("options", "ollama_start_button"), icon=icon_svg("play")),
        return ui.input_action_button("button_change_api_key", t("options", "button_change"), icon=icon_svg("wrench")),


    @reactive.effect
    @reactive.event(input.button_change_api_key)
    async def change_api_key():
        if input.api_select() == "ollama":
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
                )
            except FileNotFoundError:
                ui.modal_show(ui.modal(
                    ui.p(t("options", "ollama_start_error")),
                    title=t("options", "error_title"),
                    easy_close=True,
                    footer=ui.modal_button(t("analysis", "modal_button_close")),
                ))
                return
            ollama_status_refresh.set(ollama_status_refresh.get() + 1)

            async def _delayed_refresh():
                await asyncio.sleep(5)
                async with reactive.lock():
                    ollama_status_refresh.set(ollama_status_refresh.get() + 1)
                    await reactive.flush()
            asyncio.create_task(_delayed_refresh())
            return
        m = ui.modal(
            ui.input_password("api_key", label=None, placeholder=t("options", "add_api_key_placeholder")),
            title=t("options", "add_api_key_title"),
            easy_close=True,
            footer=(ui.input_action_button("button_save_api_key", t("options", "add_api_key_save"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"), class_="btn-danger")),
        )
        ui.modal_show(m)

    # Speichern des API-Keys
    @reactive.effect
    @reactive.event(input.button_save_api_key)
    def save_api_key():
        req(input.api_key())
        selected = input.api_select()
        if selected == "openai":
            keyring.set_password("talktrace", "api_key_openai", input.api_key())
            api_key_openai.set(input.api_key())
        elif selected == "groq":
            keyring.set_password("talktrace", "api_key_groq", input.api_key())
            api_key_groq.set(input.api_key())
        elif selected == "anthropic":
            keyring.set_password("talktrace", "api_key_anthropic", input.api_key())
            api_key_anthropic.set(input.api_key())
        elif selected == "ollama":
            keyring.set_password("talktrace", "api_key_ollama", input.api_key())
            api_key_ollama.set(input.api_key())
        ui.modal_remove()   


    @render.ui
    def loc_button_delete_api_key():
        return ui.input_action_button("button_delete_api_key", t("options", "button_delete"), icon = icon_svg("trash-can"), class_="btn-danger"),


   # Button zum Löschen des API-Keys
    @reactive.effect
    @reactive.event(input.button_delete_api_key)
    def delete_api_key():
        m = ui.modal(
            ui.p(t("options", "delete_api_key_warning")),
            title=t("options", "delete_api_key_title"),
            easy_close=True,
            footer=(ui.input_action_button("button_confirm_delete_api_key", t("options", "delete_api_key_confirm"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger")),
        )
        ui.modal_show(m)


    # Löschens des API-Keys bestätigen
    @reactive.effect
    @reactive.event(input.button_confirm_delete_api_key)
    def confirm_delete_api_key():
        selected = input.api_select()
        try:
            if selected == "openai":
                keyring.delete_password("talktrace", "api_key_openai")
            elif selected == "groq":
                keyring.delete_password("talktrace", "api_key_groq")
            elif selected == "anthropic":
                keyring.delete_password("talktrace", "api_key_anthropic")
            elif selected == "ollama":
                keyring.delete_password("talktrace", "api_key_ollama")
        except keyring.errors.PasswordDeleteError:
            pass

        if selected == "openai":
            api_key_openai.set(None)
        elif selected == "groq":
            api_key_groq.set(None)
        elif selected == "anthropic":
            api_key_anthropic.set(None)
        elif selected == "ollama":
            api_key_ollama.set(None)
        ui.modal_remove()


    # Modelle für LLM-Auswahl
    @render.ui
    def loc_llm_models():
        return ui.p(t("options", "llm_models"))
    

    # Verfügbare Modelle aus Config laden und auflisten
    @render.ui
    def loc_load_models():
        return ui.input_select("model_list", t("options", "available_models"), choices=models_available(), multiple=True)
    
    @reactive.calc
    def models_available():
        deleted_models = model_deleted.get() # for reactivity/invalidation
        return config.get_models()

    # Button zum Hinzufügen eines Modells
    @render.ui
    def loc_button_add_model():
        return ui.input_action_button("button_add_model", t("options", "add_model"), icon = icon_svg("plus"), class_="btn-success"),

    # Modal zum Hinzufügen eines Modells
    @reactive.effect
    @reactive.event(input.button_add_model)
    def add_model():
        m = ui.modal(
            ui.input_text("model_id", t("options", "model_id"), placeholder=t("options", "add_model_placeholder")),
            ui.input_select("model_provider", t("options", "model_provider"), choices=["openai", "groq", "anthropic", "ollama"], selected="openai"),
            ui.input_text("intput_cost", t("options", "input_cost"), placeholder=t("options", "cost_placeholder")),
            ui.input_text("output_cost", t("options", "output_cost"), placeholder=t("options", "cost_placeholder")),
            title=t("options", "add_model_title"),
            easy_close=True,
            footer=(ui.input_action_button("model_add_confirm", t("options", "modal_button_add"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger")),
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.model_add_confirm)
    def confirm_add_model():
        req(input.model_id(), input.model_provider())
        config.add_model(input.model_provider(), input.model_id(), float(input.intput_cost()), float(input.output_cost()))
        # Update available models in the model options
        available_models = config.get_models()
        model_deleted.set(model_deleted.get() + 1) # for reactivity/invalidation
        ui.update_select("model_list", choices=available_models)
        ui.update_select("model_select", choices=select_api_choices())
        ui.modal_remove()
    
    # Button zum Entfernen eines Modells
    @render.ui
    def loc_button_remove_model():
        return ui.input_action_button("button_remove_model", t("options", "remove_model"), icon = icon_svg("trash-can"), class_="btn-danger"),

    # Modal zum Entfernen von Modellen
    @reactive.effect
    @reactive.event(input.button_remove_model)
    def _():
        m = ui.modal(
        ui.p(t("options", "modal_remove_model_warning")),
        title=t("options", "modal_remove_title"),
        easy_close=True,
        footer=(ui.input_action_button("model_delete_confirm", t("options", "modal_remove_confirm"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger"))
        )
        ui.modal_show(m)

    # Entfernen des Modells bestätigen
    @reactive.effect
    @reactive.event(input.model_delete_confirm)
    def _():
        config.remove_model(list(input.model_list()))
        # Update available models in the model options
        model_deleted.set(model_deleted.get() + 1) # for reactivity/invalidation
        available_models = config.get_models()
        ui.update_select("model_list", choices=available_models)
        # If current selected model was removed, update model selection
        if input.model_select() not in available_models:
            ui.update_select("model_select", choices=select_api_choices())
        ui.modal_remove()


    # Modell-Auswahl auf Default zurücksetzen
    @render.ui
    def loc_button_reset_model_selection():
        return ui.input_action_button("button_reset_model_selection", t("options", "button_reset"), icon = icon_svg("arrow-rotate-left"), class_="btn-danger"),


    # Modal zum Zurücksetzen der Modellauswahl
    @reactive.effect
    @reactive.event(input.button_reset_model_selection)
    def reset_model_selection():
        m = ui.modal(
            ui.p(t("options", "reset_model_selection_confirm")),
        title=t("options", "reset_model_selection_title"),
        easy_close=True,
        footer=(ui.input_action_button("button_reset_model_selection_confirm", t("options", "modal_model_reset_confirm"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger")),
        )
        ui.modal_show(m)

    # Zurücksetzen der Modellauswahl bestätigen
    @reactive.effect
    @reactive.event(input.button_reset_model_selection_confirm)
    def confirm_reset_model_selection():
        config.reset_models()
      # ui.update_select("model_select", choices=select_api_choices())
        model_deleted.set(model_deleted.get() + 1) # for reactivity/invalidation
        ui.modal_remove()

    # Benutzerdefinierte Prompts
    @render.ui
    def loc_custom_prompts():
        return ui.p(t("options", "custom_prompts"))
    
    # System Prompt anzeigen (effektive Version inkl. Sprecher-Filter)
    @render.text()
    def system_prompt_output():
        return effective_system_prompt()
    
    # Button zum Ändern des System Prompts
    @render.ui
    def loc_button_change_system_prompt():
        return ui.input_action_button("button_change_system_prompt", t("options", "button_change"), icon = icon_svg("pen")),
    
    
    @reactive.effect
    @reactive.event(input.button_change_system_prompt)
    def change_system_prompt():
        m = ui.modal(
            ui.input_text_area("system_prompt", t("options", "change_system_prompt"), system_prompt.get(), rows=10),
            title=t("options", "change_system_prompt"),
            easy_close=True,
            footer=(ui.input_action_button("button_save_system_prompt", t("options", "add_api_key_save"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger")),
        )
        ui.modal_show(m)

    # Speichern des System Prompts
    @reactive.effect
    @reactive.event(input.button_save_system_prompt)
    def save_system_prompt():
        req(input.system_prompt())
        config.set_prompt('system', input.system_prompt())
        system_prompt.set(input.system_prompt())
        ui.modal_remove()

    
    @render.ui
    def loc_button_reset_system_prompt():
        return ui.input_action_button("button_reset_system_prompt", t("options", "button_reset"), icon = icon_svg("arrow-rotate-left"), class_="btn-danger"),


    # Button zum Zurücksetzen des System Prompts
    @reactive.effect
    @reactive.event(input.button_reset_system_prompt)
    def reset_system_prompt():
        m = ui.modal(
            ui.p(t("options", "reset_system_prompt_confirm")),
        title=t("options", "reset_system_prompt_title"),
        easy_close=True,
        footer=(ui.input_action_button("button_reset_system_prompt_confirm", t("analysis", "modal_confirm_reset"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger")),
        )
        ui.modal_show(m)

    # Zurücksetzen des System Prompts Bestätigen
    @reactive.effect
    @reactive.event(input.button_reset_system_prompt_confirm)
    def confirm_reset_system_prompt():
        config.set_prompt('system', config.get_prompts()['system_default'])
        system_prompt.set(config.get_prompts()['system'])
        ui.modal_remove()


    # User Prompt anzeigen (aktuell ohne Sprecher-Filter, aber via effective_*-Getter
    # konsistent gehalten, falls später zusätzlich angepasst werden soll)
    @render.text()
    def user_prompt_output():
        return effective_user_prompt()
    

    @render.ui
    def loc_button_change_user_prompt():
        return ui.input_action_button("button_change_user_prompt", t("options", "button_change"), icon = icon_svg("pen")),


    # Button zum Ändern des User Prompts
    @reactive.effect
    @reactive.event(input.button_change_user_prompt)
    def change_user_prompt():
        m = ui.modal(
            ui.input_text_area("user_prompt", t("options", "change_user_prompt"), user_prompt.get(), rows=10),
            title=t("options", "change_user_prompt"),
            easy_close=True,
            footer=(ui.input_action_button("button_save_user_prompt", t("options", "add_api_key_save"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"), class_="btn-danger")),
        )
        ui.modal_show(m)


    # Speichern des User Prompts
    @reactive.effect
    @reactive.event(input.button_save_user_prompt)
    def save_user_prompt():
        req(input.user_prompt())
        config.set_prompt('user', input.user_prompt())
        user_prompt.set(input.user_prompt())
        ui.modal_remove()   


    # Button zum Zurücksetzen des User Prompts
    @render.ui
    def loc_button_reset_user_prompt():
        return ui.input_action_button("button_reset_user_prompt", t("options", "button_reset"), icon = icon_svg("arrow-rotate-left"), class_="btn-danger"),


    @reactive.effect
    @reactive.event(input.button_reset_user_prompt)
    def reset_user_prompt():
        m = ui.modal(
            ui.p(t("options", "reset_user_prompt_confirm")),
        title=t("options", "reset_user_prompt_title"),
        easy_close=True,
        footer=(ui.input_action_button("button_reset_user_prompt_confirm", t("analysis", "modal_confirm_reset"), class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"), class_="btn-danger")),
        )
        ui.modal_show(m)

    # Zurücksetzen des User Prompts Bestätigen
    @reactive.effect
    @reactive.event(input.button_reset_user_prompt_confirm)
    def confirm_reset_user_prompt():
        config.set_prompt('user', config.get_prompts()['user_default'])
        user_prompt.set(config.get_prompts()['user'])
        ui.modal_remove()

    # Weitere Optionen
    @render.ui
    def loc_additional_options():
        return ui.p(t("options", "additional_options"))
    

    @render.ui
    def loc_input_teacher_name_options():
        return ui.input_text("name_teacher_options", t("options", "teacher_name"), config.get_parameters()['teacher_name'])
    
    # Parameter in Config Speichern
    @reactive.effect
    @reactive.event(input.name_teacher_options)
    def _():
        config.set_parameter('teacher_name', input.name_teacher_options())


    @render.ui
    def loc_input_group_id_options():
        return ui.input_text("name_group_options", t("options", "group_id"), "B1")
    
    @reactive.effect
    @reactive.event(input.name_group_options)
    def _():
        config.set_parameter('group_id', input.name_group_options())


    @render.ui
    def loc_input_num_pupils_options():
        return ui.input_numeric("num_pupils_options", t("options", "num_students"), 25, min=1, max=100)


    @reactive.effect
    @reactive.event(input.num_pupils_options)
    def _():
        config.set_parameter('num_pupils', input.num_pupils_options())


    @render.ui
    def loc_button_reset_parameters():
        return ui.input_action_button("button_reset_parameters", t("options", "button_reset"), icon = icon_svg("arrow-rotate-left"), class_="btn-danger"),
    

    # Button zum Zurücksetzen der Gruppen-Parameter
    @reactive.effect
    @reactive.event(input.button_reset_parameters)
    def reset_user_prompt():
        m = ui.modal(
            ui.p(t("options", "reset_group_parameters_confirm")),
        title=t("options", "reset_group_parameters_title"),
        easy_close=True,
        footer=(ui.input_action_button("button_reset_parameters_confirm", t("analysis", "modal_confirm_reset"), class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"), class_="btn-danger")),
        )
        ui.modal_show(m)

    # Zurücksetzen der Gruppen-Parameter bestätigen
    @reactive.effect
    @reactive.event(input.button_reset_parameters_confirm)
    def confirm_reset_parameters():
        config.set_parameter('teacher_name', config.get_parameters()['teacher_name_default'])
        config.set_parameter('group_id', config.get_parameters()['group_id_default'])
        config.set_parameter('num_pupils', config.get_parameters()['num_pupils_default'])
        ui.update_text("name_teacher_options", value=config.get_parameters()['teacher_name'])
        ui.update_text("name_group_options", value=config.get_parameters()['group_id'])
        ui.update_numeric("num_pupils_options", value=config.get_parameters()['num_pupils'])
        ui.modal_remove()

    # About TalkTrace AI
    @render.ui
    def loc_app_info():
        return ui.p(t("options", "about"))
    
    @render.ui
    def loc_app_info_text():
        return ui.markdown(t("options", "about_text"))

# -----------------------------------------------------------------------------------------------------------   

# App als globales Objekt initiasieren, damit der server zugreifen kann
app = App(app_ui, server, debug=False)

# Get the directory containing the current file
current_dir = Path(__file__).parent

def main(open_window: bool = True):
    host, port = "127.0.0.1", 8000

    if not open_window:
        run_app(app, host=host, port=port, launch_browser=False)
        return

    import threading
    import time
    import socket
    import webview

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

    webview.create_window(
        "TalkTrace AI",
        f"http://{host}:{port}",
        width=1280,
        height=860,
    )
    webview.start()
