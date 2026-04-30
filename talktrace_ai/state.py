from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from shiny import reactive

from .config.config_manager import ConfigManager
from .localization.translation import TRANSLATIONS


@dataclass
class AppState:
    input: Any
    output: Any
    session: Any
    config: ConfigManager
    t: Callable[[str, str], str]

    transcript_data: Any
    transcript_format_status: Any
    codebook_data: Any
    converted_transcript: Any
    fmt_text: Any
    fmt_analysis: Any
    fmt_options: Any
    fmt_meta: Any
    api_key_groq: Any
    api_key_openai: Any
    api_key_anthropic: Any
    api_key_ollama: Any
    ollama_status_refresh: Any
    current_api: Any
    num_participants: Any
    participation_rate: Any
    t_turns: Any
    t_turns_length: Any
    t_turns_length_mean_sd: Any
    p_turns: Any
    p_turns_length: Any
    p_turns_length_mean_sd: Any
    stats: Any
    stats_per_speaker: Any
    llm_analysis_data: Any
    model: Any
    teacher_impulses_count: Any
    analysis_state: Any
    analysis_llm_state: Any
    sim_plot: Any
    qual_plot: Any
    qual_stats_df: Any
    placeholder_plot: Any
    model_deleted: Any
    current_lang: Any
    code_legend_storage: Any
    estimated_cost: Any
    token_count: Any
    report_a_df: Any
    report_b_df: Any
    report_a_error: Any
    report_b_error: Any
    expert_mode_on: Any
    expert_metric: Any
    expert_n_raters: Any
    expert_result: Any
    expert_error: Any
    system_prompt: Any
    user_prompt: Any
    autopilot_running: Any
    autopilot_phase: Any
    autopilot_results: Any
    autopilot_error: Any
    autopilot_make_reports: Any
    autopilot_report_format: Any
    autopilot_report_a_path: Any
    autopilot_report_b_path: Any
    autopilot_report_a_pending: Any
    autopilot_report_b_pending: Any
    autopilot_report_a_error: Any
    autopilot_report_b_error: Any
    tab_badge_results: Any
    tab_badge_testing: Any

    run_analysis: Optional[Callable[..., Any]] = None
    select_api_choices: Optional[Callable[..., Any]] = None
    effective_system_prompt: Optional[Callable[..., Any]] = None
    effective_user_prompt: Optional[Callable[..., Any]] = None
    make_sim_stats_over_time_plot: Optional[Callable[..., Any]] = None
    make_qualitative_stats_over_time_plot: Optional[Callable[..., Any]] = None
    segment_labels_for: Optional[Callable[..., Any]] = None


def build_app_state(input, output, session) -> AppState:
    config = ConfigManager()
    current_lang = reactive.value(config.get_localization()["current_language"])

    def t(section, key):
        return TRANSLATIONS[current_lang.get()][section][key]

    return AppState(
        input=input,
        output=output,
        session=session,
        config=config,
        t=t,
        transcript_data=reactive.value(None),
        transcript_format_status=reactive.value(None),
        codebook_data=reactive.value(None),
        converted_transcript=reactive.value(None),
        fmt_text=reactive.value(None),
        fmt_analysis=reactive.value(None),
        fmt_options=reactive.value(None),
        fmt_meta=reactive.value(None),
        api_key_groq=reactive.value(),
        api_key_openai=reactive.value(),
        api_key_anthropic=reactive.value(),
        api_key_ollama=reactive.value(),
        ollama_status_refresh=reactive.value(0),
        current_api=reactive.value(config.get_current_api()),
        num_participants=reactive.value(None),
        participation_rate=reactive.value(None),
        t_turns=reactive.value(None),
        t_turns_length=reactive.value(None),
        t_turns_length_mean_sd=reactive.value(None),
        p_turns=reactive.value(None),
        p_turns_length=reactive.value(None),
        p_turns_length_mean_sd=reactive.value(None),
        stats=reactive.value(None),
        stats_per_speaker=reactive.value(None),
        llm_analysis_data=reactive.value([]),
        model=reactive.value(config.get_current_model()),
        teacher_impulses_count=reactive.value(None),
        analysis_state=reactive.value(False),
        analysis_llm_state=reactive.value(False),
        sim_plot=reactive.value(None),
        qual_plot=reactive.value(),
        qual_stats_df=reactive.value(None),
        placeholder_plot=reactive.value(),
        model_deleted=reactive.value(0),
        current_lang=current_lang,
        code_legend_storage=reactive.value("Legende nicht ausgelesen"),
        estimated_cost=reactive.value(None),
        token_count=reactive.value(None),
        report_a_df=reactive.value(None),
        report_b_df=reactive.value(None),
        report_a_error=reactive.value(None),
        report_b_error=reactive.value(None),
        expert_mode_on=reactive.value(False),
        expert_metric=reactive.value("cohen"),
        expert_n_raters=reactive.value(2),
        expert_result=reactive.value(None),
        expert_error=reactive.value(None),
        system_prompt=reactive.value(config.get_prompts()['system']),
        user_prompt=reactive.value(config.get_prompts()['user']),
        autopilot_running=reactive.value(False),
        autopilot_phase=reactive.value(None),
        autopilot_results=reactive.value({}),
        autopilot_error=reactive.value(None),
        autopilot_make_reports=reactive.value(False),
        autopilot_report_format=reactive.value("docx"),
        autopilot_report_a_path=reactive.value(None),
        autopilot_report_b_path=reactive.value(None),
        autopilot_report_a_pending=reactive.value(False),
        autopilot_report_b_pending=reactive.value(False),
        autopilot_report_a_error=reactive.value(None),
        autopilot_report_b_error=reactive.value(None),
        tab_badge_results=reactive.value(None),
        tab_badge_testing=reactive.value(None),
    )
