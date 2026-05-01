"""Backwards-compatible re-export of helpers that used to live
directly in this file. The real definitions are now in
``talktrace_ai/utils/*``; this shim keeps existing imports valid.
"""
from .utils._config import _get_config, translate
from .utils.credentials import (
    safe_get_password, safe_set_password, safe_delete_password,
    keyring_available, _keyring_unavailable,
)
from .utils.llm_cache import _cache_key, _cache_get, _cache_put
from .utils.llm_clients import (
    get_groq_client, get_openai_client, get_anthropic_client,
)
from .utils.file_io import (
    docx_to_json, read_txt, import_file, write_txt, write_docx_from_text,
)
from .utils.intercoder import (
    parse_report_impulses, compute_intercoder_agreement, export_testing_agreement,
    export_testing_agreement_any, compute_intercoder_agreement_multi,
    p_value_stars,
)
from .utils.transcript_format import (
    is_valid_transcript_format, convert_to_standard_format,
)
from .utils.stats import (
    count_pupils, dialog_stats, dialog_stats_per_speaker,
    dialog_stats_over_time, map_impulses_to_turn_index,
    code_distribution_over_time, count_transcript_turns,
    count_teacher_impulses, _parse_turns,
)
from .utils.history import (
    save_to_history, list_history, load_history_entry, delete_history_entry,
)
from .utils.llm_analysis import (
    llm_analysis_groq, llm_analysis_openai,
    llm_analysis_anthropic, llm_analysis_ollama,
    llm_analysis_groq_stream, llm_analysis_openai_stream,
    llm_analysis_anthropic_stream, llm_analysis_ollama_stream,
    async_stream,
)
from .utils.reports import (
    generate_report2, DEFAULT_REPORT_SECTIONS,
)
from .utils.cost_tracker import (
    record_run as record_cost_run,
    get_summary as get_cost_summary,
    reset_log as reset_cost_log,
)
from .utils.fingerprint import compute_fingerprint
from .utils.self_test import run_self_test
