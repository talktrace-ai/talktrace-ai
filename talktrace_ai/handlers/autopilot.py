"""Autopilot orchestrator: code once with LLM A, then with LLM B, then push
both results into the Testing tab as Coder A / Coder B.

The two coding runs are sequential, not parallel — sequential keeps
provider rate-limits and progress-feedback simple, and prevents the two
LLMs from competing for shared reactive UI state. Streaming is *not* used
here even when ``advanced.streaming`` is enabled in config: the autopilot
exposes its own coarse "Step 1/2 → Step 2/2" progress, which would fight
with the streaming stepper if both were active.
"""
from ._common import *

from ..utils.llm_analysis._core import build_effective_prompts, run_llm_coding_once


def _api_key_for(state, provider: str):
    if provider == "openai":
        return state.api_key_openai.get()
    if provider == "groq":
        return state.api_key_groq.get()
    if provider == "anthropic":
        return state.api_key_anthropic.get()
    return None  # ollama: keyless


def _speaker_flags_from_mode(mode: str):
    """Map the Autopilot's three-way radio into (teacher_on, students_on)."""
    if mode == "teacher":
        return True, False
    if mode == "students":
        return False, True
    return True, True


def _to_report_df(df: pd.DataFrame) -> pd.DataFrame:
    """Project a coding DataFrame to the column order the Testing tab expects."""
    if df is None:
        return None
    cols = ["#", "Sprecher", "Impuls", "Shortcode"]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    return out[cols]


def _save_autopilot_pickle(state, df: pd.DataFrame, model: str, suffix: str) -> None:
    """Persist a single coder's run as a session pickle in the history dir.

    Filename pattern reuses ``save_to_history`` (timestamp_group_model.pkl);
    we encode the coder slot as a model-name suffix so both files are easy
    to spot in the history list.
    """
    input = state.input
    config = state.config

    session_data = {
        "transcript_data": state.transcript_data.get(),
        "num_participants": state.num_participants.get(),
        "participation_rate": state.participation_rate.get(),
        "stats": state.stats.get(),
        "llm_analysis_data": [df],
        "analysis_llm_state": True,
        "code_legend_storage": state.code_legend_storage.get(),
    }
    try:
        n_turns = int(state.stats.get()['Anzahl_Beitraege'].sum()) if state.stats.get() is not None else 0
    except Exception:
        n_turns = 0
    try:
        group_id = input.autopilot_name_group() or ""
    except Exception:
        group_id = ""
    try:
        save_to_history(
            session_data,
            group_id=group_id,
            model=f"{model}_{suffix}",
            n_turns=n_turns,
            n_pupils=state.num_participants.get(),
            participation_rate=state.participation_rate.get(),
            language=config.get_localization().get("current_language"),
        )
    except Exception as exc:
        print(f"[autopilot] history save failed for {suffix}: {exc}")


def _autopilot_teacher_name(state) -> str:
    """Read the teacher name from the autopilot's own input, falling back to
    the placeholder used by the manual flow."""
    try:
        name = state.input.autopilot_name_teacher()
    except Exception:
        name = None
    return (name or state.t("analysis", "name_teacher_var")).strip()


def _teacher_in_transcript(transcript: str, teacher_name: str) -> bool:
    """Mirror the manual-flow check: speaker-label match (``Name:``) or
    plain substring match. Empty/None inputs return False so the caller
    can surface a helpful error instead of silently coding nothing."""
    if not teacher_name or transcript is None:
        return False
    search_name = re.escape(teacher_name)
    pattern = re.compile(rf"^\s*{search_name}\s*:", re.IGNORECASE | re.MULTILINE)
    if pattern.search(transcript):
        return True
    return teacher_name.lower() in transcript.lower()


async def _compute_quant_stats(state, *, teacher_name: str, num_class: int):
    """Run the same quantitative pipeline the manual analysis flow runs.

    The manual flow (``handlers/sidebar/_analysis.py``) computes per-speaker
    turn stats + participation rate alongside the LLM call so the Results
    tab can render them. The autopilot used to skip this entirely, which
    meant: (1) the autopilot's history pickle was missing ``stats`` and the
    derived turn/length values, so reloading the pickle later crashed the
    quantitative panel with ``'NoneType' object has no attribute 'plot'``,
    and (2) the autopilot tab itself couldn't surface anything beyond the
    codings. We compute stats once at the start of the autopilot run, before
    Coder A — both codings share the same transcript so the stats are
    identical for both pickles.
    """
    transcript = state.transcript_data.get()

    def _do():
        return {
            "num_participants": count_pupils(transcript),
            "stats": dialog_stats(transcript, teacher_name),
            "stats_per_speaker": dialog_stats_per_speaker(transcript, teacher_name),
        }

    result = await asyncio.to_thread(_do)
    df_stats = result["stats"]

    def _safe(speaker, col, default=0):
        if df_stats is None:
            return default
        m = df_stats.loc[df_stats['Sprecher'] == speaker, col]
        return m.values[0] if not m.empty else default

    num_p = result["num_participants"] or 0

    async with reactive.lock():
        state.num_participants.set(result["num_participants"])
        state.stats.set(df_stats)
        state.stats_per_speaker.set(result["stats_per_speaker"])
        state.teacher_impulses_count.set(count_teacher_impulses(df_stats, teacher_name))
        state.participation_rate.set((num_p / num_class * 100) if num_class else 0)
        state.t_turns.set(_safe(teacher_name, 'Anzahl_Beitraege'))
        state.t_turns_length.set(round(_safe(teacher_name, 'Durchschnitt_Woerter'), 1))
        state.t_turns_length_mean_sd.set(round(_safe(teacher_name, 'Median_Woerter'), 1))
        state.p_turns.set(_safe("Schüler:innen", 'Anzahl_Beitraege'))
        state.p_turns_length.set(round(_safe("Schüler:innen", 'Durchschnitt_Woerter'), 1))
        state.p_turns_length_mean_sd.set(_safe("Schüler:innen", 'Median_Woerter'))
        await reactive.flush()


async def _do_coding(state, *, provider: str, model: str, multi_coding: bool,
                     teacher_on: bool, students_on: bool):
    """Build prompts + call the pure core. Returns (df, raw, err)."""
    transcript = state.transcript_data.get()
    codebook = state.codebook_data.get()
    teacher_name = _autopilot_teacher_name(state)

    sys_p, usr_p = build_effective_prompts(
        state.system_prompt.get(),
        state.user_prompt.get(),
        t=state.t,
        teacher_on=teacher_on,
        students_on=students_on,
        multi_coding=multi_coding,
        teacher_name=teacher_name,
    )

    return await asyncio.to_thread(
        run_llm_coding_once,
        provider=provider,
        model=model,
        transcript=transcript,
        codebook=codebook,
        system_prompt=sys_p,
        user_prompt=usr_p,
        api_key=_api_key_for(state, provider),
    )


_PROVIDER_LABELS = {"openai": "OpenAI", "groq": "Groq", "anthropic": "Anthropic", "ollama": "Ollama"}


def _quant_summary_block(state, t):
    """Render the collapsible quantitative-stats summary that lives under the
    side-by-side coder tables. Pulls from ``state.stats_per_speaker`` etc.
    that ``_compute_quant_stats`` populates at the start of the run.

    Returned as a plain UI fragment (not a ``render.ui``) because it sits
    inside the parent ``loc_autopilot_results_section`` and re-renders
    together with the codings whenever ``autopilot_phase`` flips to "done".
    """
    stats_per_speaker = state.stats_per_speaker.get()
    num_part = state.num_participants.get()
    part_rate = state.participation_rate.get()
    t_turns = state.t_turns.get()
    p_turns = state.p_turns.get()

    if stats_per_speaker is None and num_part is None:
        return None  # quant computation didn't run for some reason — stay silent

    def _box(label, value):
        return ui.value_box(
            ui.tags.span(label, style="font-size:0.85rem;"),
            ui.tags.span(str(value if value is not None else "—"),
                         style="font-size:1.4rem;"),
            theme="secondary",
        )

    boxes = ui.layout_columns(
        _box(t("autopilot", "quant_box_participants"), num_part),
        _box(t("autopilot", "quant_box_participation_rate"),
             f"{part_rate:.1f} %" if isinstance(part_rate, (int, float)) else "—"),
        _box(t("autopilot", "quant_box_teacher_turns"), t_turns),
        _box(t("autopilot", "quant_box_student_turns"), p_turns),
    )

    if stats_per_speaker is not None and len(stats_per_speaker) > 0:
        # Round float columns to one decimal place so the table reads cleanly.
        df_display = stats_per_speaker.copy()
        for col in ("Durchschnitt_Woerter", "Median_Woerter"):
            if col in df_display.columns:
                df_display[col] = pd.to_numeric(df_display[col], errors="coerce").round(1)
        # Localize raw DataFrame column names (Sprecher / Anzahl_Beitraege / …)
        # to user-facing labels — without this, English users see the German
        # internal names and umlauts get encoded as "ae"/"oe".
        df_display = df_display.rename(columns={
            "Sprecher": t("autopilot", "quant_col_speaker"),
            "Anzahl_Beitraege": t("autopilot", "quant_col_turns"),
            "Gesamt_Woerter": t("autopilot", "quant_col_words_total"),
            "Durchschnitt_Woerter": t("autopilot", "quant_col_words_avg"),
            "Median_Woerter": t("autopilot", "quant_col_words_median"),
        })
        table_html = df_display.to_html(
            index=False, classes="table table-sm table-striped",
            border=0, escape=True,
        )
        # Pandas hard-codes ``style="text-align: right;"`` on the thead row,
        # which makes column titles (Sprecher, Anzahl_Beitraege, …) sit
        # right-aligned over left-aligned data cells — the column header
        # "Sprecher" then visually floats far to the right of S01/S02/…
        # Strip that style so headers align with the data underneath.
        table_html = table_html.replace(' style="text-align: right;"', "")
        table_block = ui.div(
            ui.HTML(table_html),
            style="max-height: 360px; overflow-y: auto; font-size: 0.92rem; text-align: left;",
        )
    else:
        table_block = ui.p("—")

    return ui.accordion(
        ui.accordion_panel(
            t("autopilot", "quant_panel_title"),
            ui.p(t("autopilot", "quant_panel_intro"), class_="text-muted"),
            boxes,
            ui.tags.h6(t("autopilot", "quant_table_title"),
                       style="margin-top:1rem; text-align:left;"),
            table_block,
            value="autopilot_quant",
            icon=icon_svg("chart-column"),
        ),
        id="autopilot_quant_accordion",
        open=False,
    )


def register(state):
    input = state.input
    t = state.t
    config = state.config

    autopilot_running = state.autopilot_running
    autopilot_phase = state.autopilot_phase
    autopilot_results = state.autopilot_results
    autopilot_error = state.autopilot_error
    report_a_df = state.report_a_df
    report_b_df = state.report_b_df
    report_a_error = state.report_a_error
    report_b_error = state.report_b_error

    # ---- UI renderers --------------------------------------------------

    @render.text
    def loc_title_autopilot():
        return t("autopilot", "tab_title")

    @render.ui
    def loc_autopilot_intro():
        return ui.p(t("autopilot", "intro_header"))

    @render.ui
    def loc_autopilot_intro_body():
        return ui.p(t("autopilot", "intro_body"))

    @render.ui
    def loc_autopilot_inputs_header():
        return ui.p(t("autopilot", "inputs_header"))

    @render.ui
    def loc_autopilot_upload_transcript():
        # Same accept-list and same reactive target (state.transcript_data)
        # as the Analysis tab so uploads are interchangeable. The Analysis
        # handler's @reactive.event(input.transcript) listener writes the
        # parsed file into transcript_data; we mirror that here.
        return ui.div(
            ui.div(
                ui.input_file(
                    "autopilot_transcript",
                    t("autopilot", "upload_transcript"),
                    multiple=False,
                    accept=[".txt", ".docx"],
                    button_label=t("analysis", "browse"),
                    placeholder=t("analysis", "placeholder"),
                ),
                class_="ttai-file-wrap",
                style="flex: 0 1 auto; min-width: 0;",
            ),
            ui.div(
                ui.output_ui("loc_transcript_format_status_autopilot"),
                style="flex: 0 0 auto; align-self: flex-end; display: inline-flex; align-items: center;",
            ),
            ui.div(
                ui.tooltip(
                    ui.input_action_button(
                        "button_check_format_autopilot",
                        "",
                        icon=icon_svg("wand-magic-sparkles"),
                        class_="btn-default",
                        style="width: 1.875rem; height: 1.875rem; padding: 0; display: inline-flex; align-items: center; justify-content: center; line-height: 1;",
                    ),
                    t("analysis", "check_format_tooltip"),
                    placement="right",
                ),
                style="flex: 0 0 auto; align-self: flex-end;",
            ),
            style="display: flex; gap: 0.5rem; align-items: start;",
        )

    @render.ui
    def loc_autopilot_upload_codebook():
        return ui.input_file(
            "autopilot_codebook",
            t("autopilot", "upload_codebook"),
            multiple=False,
            accept=[".txt", ".docx"],
            button_label=t("analysis", "browse"),
            placeholder=t("analysis", "placeholder"),
        )

    @render.ui
    def loc_autopilot_inputs_status():
        items = []
        transcript_loaded = state.transcript_data.get() is not None
        codebook_loaded = state.codebook_data.get() is not None
        # Daten können auch über den Analyse-Tab geladen sein. Der Datei-
        # Widget hier ist dann leer, der gemeinsame State aber gefüllt —
        # Quelle annotieren, damit der "geladen"-Status nicht verwirrt.
        try:
            from_autopilot_t = bool(input.autopilot_transcript())
        except Exception:
            from_autopilot_t = False
        try:
            from_autopilot_c = bool(input.autopilot_codebook())
        except Exception:
            from_autopilot_c = False
        from_analysis_t = transcript_loaded and not from_autopilot_t
        from_analysis_c = codebook_loaded and not from_autopilot_c

        def _status_line(loaded, loaded_key, missing_key, from_analysis):
            label = t("autopilot", loaded_key) if loaded else t("autopilot", missing_key)
            children = ["✓ " if loaded else "○ ", label]
            if from_analysis:
                children.append(ui.tags.span(
                    " " + t("autopilot", "status_source_analysis"),
                    class_="text-muted",
                    style="font-size: 0.85em;",
                ))
            return ui.tags.div(*children, class_="text-success" if loaded else "text-muted")

        items.append(_status_line(
            transcript_loaded, "status_transcript_loaded",
            "status_transcript_missing", from_analysis_t,
        ))
        items.append(_status_line(
            codebook_loaded, "status_codebook_loaded",
            "status_codebook_missing", from_analysis_c,
        ))
        return ui.div(*items, style="margin-top:0.5rem;")

    @render.ui
    def loc_autopilot_general_header():
        return ui.p(t("autopilot", "general_header"))

    @render.ui
    def loc_autopilot_group_id():
        return ui.input_text(
            "autopilot_name_group",
            t("analysis", "group_id"),
            config.get_parameters()["group_id"],
        )

    @render.ui
    def loc_autopilot_num_pupils():
        return ui.input_numeric(
            "autopilot_num_pupils",
            t("analysis", "num_pupils"),
            config.get_parameters()["num_pupils"],
            min=1, max=100,
        )

    @render.ui
    def loc_autopilot_name_teacher():
        return ui.input_text(
            "autopilot_name_teacher",
            t("analysis", "name_teacher"),
            config.get_parameters()["teacher_name"],
        )

    @render.ui
    def loc_autopilot_options_header():
        return ui.p(t("autopilot", "options_header"))

    @render.ui
    def loc_autopilot_options():
        return ui.div(
            ui.input_switch(
                "autopilot_multi_coding",
                t("sidebar", "multi_coding_switch"),
                False,
            ),
            ui.input_radio_buttons(
                "autopilot_speaker_mode",
                t("autopilot", "speaker_mode_label"),
                choices={
                    "both": t("autopilot", "speaker_mode_both"),
                    "teacher": t("autopilot", "speaker_mode_teacher"),
                    "students": t("autopilot", "speaker_mode_students"),
                },
                selected="both",
                inline=True,
            ),
        )

    @render.ui
    def loc_autopilot_coders_header():
        return ui.p(t("autopilot", "coders_header"))

    def _coder_block(slot: str):
        # slot is "a" or "b" — drives input ids and labels.
        provider_id = f"autopilot_provider_{slot}"
        model_id = f"autopilot_model_{slot}"
        try:
            current_provider = input[provider_id]() or "openai"
        except Exception:
            current_provider = "openai"
        models = config.get_models(provider=current_provider) or {}
        return ui.div(
            ui.h5(t("autopilot", f"coder_{slot}_label")),
            ui.input_select(
                provider_id,
                t("sidebar", "provider_select"),
                choices=_PROVIDER_LABELS,
                selected=current_provider,
            ),
            ui.input_select(
                model_id,
                t("sidebar", "model_select"),
                choices=models,
            ),
            class_="ttai-autopilot-coder",
        )

    @render.ui
    def loc_autopilot_coder_a():
        return _coder_block("a")

    @render.ui
    def loc_autopilot_coder_b():
        return _coder_block("b")

    def _selected_pair():
        try:
            pa = input.autopilot_provider_a()
            ma = input.autopilot_model_a()
            pb = input.autopilot_provider_b()
            mb = input.autopilot_model_b()
        except Exception:
            return None, None, None, None
        return pa, ma, pb, mb

    def _is_same_pair():
        pa, ma, pb, mb = _selected_pair()
        if not (pa and ma and pb and mb):
            return False
        return (pa, ma) == (pb, mb)

    @render.ui
    def loc_autopilot_validation():
        if not _is_same_pair():
            return None
        return ui.div(
            icon_svg("triangle-exclamation"),
            " ", t("autopilot", "warning_same_model"),
            class_="text-warning",
            style="margin-top:0.5rem; font-size:0.95rem;",
        )

    @render.ui
    def loc_autopilot_start_button():
        running = autopilot_running.get()
        same_pair = _is_same_pair()
        button = ui.input_action_button(
            "autopilot_start",
            t("autopilot", "start_button"),
            icon=icon_svg("plane-departure"),
            class_="btn-success",
            disabled=running or same_pair,
        )
        return ui.div(
            ui.div(button, style="flex: 0 0 auto;"),
            ui.div(
                ui.output_ui("loc_autopilot_cost_chip"),
                style="flex: 0 0 auto;",
            ),
            style="display:flex; align-items:center; justify-content:flex-start; gap:0.5rem; margin-top:0.5rem;",
        )

    def _format_cost(cost: float, lang: str) -> str:
        s = f"{cost:.2f}"
        return s.replace(".", ",") if lang == "de" else s

    def _cost_for_pair(tokens: int, provider: str, model: str):
        pricing = config.get_api_pricing()
        if provider in pricing and model in pricing[provider]:
            rate_in = pricing[provider][model]["input"]
            rate_out = pricing[provider][model]["output"]
            return (tokens / 1_000_000) * rate_in + (tokens / 1_000_000) * rate_out * 4
        return None

    @render.ui
    def loc_autopilot_cost_chip():
        try:
            lang = state.current_lang.get()
        except Exception:
            lang = "en"

        transcript = state.transcript_data.get()
        codebook = state.codebook_data.get() or ""
        pa, ma, pb, mb = _selected_pair()

        total = None
        tokens = 0
        if transcript and pa and ma and pb and mb:
            try:
                sys_p, usr_p = build_effective_prompts(
                    state.system_prompt.get(),
                    state.user_prompt.get(),
                    t=t,
                    teacher_on=True, students_on=True,
                    multi_coding=False,
                    teacher_name=_autopilot_teacher_name(state),
                )
                encoding = tiktoken.get_encoding("cl100k_base")
                all_text = f"{sys_p}\n{usr_p}\n{transcript}\n{codebook}"
                tokens = len(encoding.encode(all_text))
            except Exception as exc:
                print(f"[autopilot cost] token calc failed: {exc}")
                tokens = 0

            if tokens:
                ca = _cost_for_pair(tokens, pa, ma)
                cb = _cost_for_pair(tokens, pb, mb)
                if ca is not None and cb is not None:
                    total = ca + cb

        if total is not None:
            amount = f"≈ {_format_cost(total, lang)} €"
        else:
            amount = f"{_format_cost(0.0, lang)} €"
        tooltip = t("sidebar", "cost_prediction")
        if tokens:
            tooltip = f"{tooltip} · {t('sidebar', 'tokens_aprox')} {tokens:,} (A+B)"
        return ui.div(
            icon_svg("coins"),
            ui.span(amount, class_="ttai-cost-chip__amount"),
            class_="ttai-cost-chip ttai-cost-chip--inline",
            title=tooltip,
        )

    def _df_to_table(df):
        """Render a coding DataFrame as a scrollable HTML table.

        We don't use ``render.data_frame`` here because the two tables sit
        inside a ``layout_columns`` and we need a single ``output_ui``
        switch to hide the whole results section before the run is done.
        """
        if df is None or len(df) == 0:
            return ui.p("—")
        # DataFrame.to_html escapes by default — safe to wrap in ui.HTML.
        html = df.to_html(index=False, classes="table table-sm table-striped",
                          border=0, escape=True)
        return ui.div(
            ui.HTML(html),
            style="max-height: 480px; overflow-y: auto; font-size: 0.92rem;",
        )

    @render.ui
    def loc_results_autopilot_banner():
        # Dezenter Hinweis auf dem Results-Tab: dorthin schreibt der Autopilot
        # bewusst NICHT (Results zeigt nur die manuelle Einzel-Analyse), sonst
        # würde die Auto-Save-/Auto-Switch-Logik der manuellen Analyse mit
        # dem Autopilot-State kollidieren. Stattdessen verweisen wir auf den
        # Autopilot-Tab für die zwei Codierungen und auf Testing für κ.
        if autopilot_phase.get() != "done":
            return None
        return ui.div(
            icon_svg("circle-info"),
            " ", t("autopilot", "results_banner_text"),
            class_="alert alert-info",
            style="margin: 0.5rem 0; padding: 0.6rem 0.9rem; font-size: 0.92rem;",
        )

    @render.ui
    def loc_autopilot_results_section():
        if autopilot_phase.get() != "done":
            return None
        results = autopilot_results.get() or {}
        a = results.get("a")
        b = results.get("b")
        if not (a and b):
            return None
        return ui.card(
            ui.card_header(ui.p(t("autopilot", "results_header"))),
            ui.p(t("autopilot", "results_intro"), class_="text-muted"),
            ui.layout_columns(
                ui.card(
                    ui.card_header(f"Coder A — {a.get('model', '')}"),
                    _df_to_table(a.get("df")),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header(f"Coder B — {b.get('model', '')}"),
                    _df_to_table(b.get("df")),
                    full_screen=True,
                ),
            ),
            _quant_summary_block(state, t),
        )

    @render.ui
    def loc_autopilot_progress():
        phase = autopilot_phase.get()
        err = autopilot_error.get()
        results = autopilot_results.get() or {}

        if phase is None and not err:
            return None

        items = []
        # Step indicators
        def step(label_key, status):
            # status: "done" | "running" | "failed" | "pending"
            color = {
                "done": "text-success",
                "running": "text-primary",
                "failed": "text-danger",
                "pending": "text-muted",
            }[status]
            icon_name = {
                "done": "circle-check",
                "running": "spinner",
                "failed": "circle-xmark",
                "pending": "circle",
            }[status]
            return ui.tags.div(
                icon_svg(icon_name),
                " ", t("autopilot", label_key),
                class_=color,
                style="font-size: 1.05rem; margin-bottom: 0.25rem;",
            )

        if phase == "coder_a_running":
            items.append(step("step_coder_a", "running"))
            items.append(step("step_coder_b", "pending"))
        elif phase == "coder_a_failed":
            items.append(step("step_coder_a", "failed"))
            items.append(step("step_coder_b", "pending"))
        elif phase == "coder_b_running":
            items.append(step("step_coder_a", "done"))
            items.append(step("step_coder_b", "running"))
        elif phase == "coder_b_failed":
            items.append(step("step_coder_a", "done"))
            items.append(step("step_coder_b", "failed"))
        elif phase == "done":
            items.append(step("step_coder_a", "done"))
            items.append(step("step_coder_b", "done"))
            items.append(ui.tags.div(
                t("autopilot", "success_done"),
                class_="text-success",
                style="margin-top: 0.5rem; font-weight: 600;",
            ))

        if err:
            items.append(ui.tags.div(
                f"{t('autopilot', 'error_prefix')}: {err}",
                class_="text-danger",
                style="margin-top: 0.5rem;",
            ))

        if phase == "coder_b_failed" and results.get("a"):
            items.append(ui.input_action_button(
                "autopilot_retry_b",
                t("autopilot", "retry_b_button"),
                icon=icon_svg("rotate-right"),
                class_="btn-warning",
                disabled=autopilot_running.get(),
            ))

        return ui.div(*items, style="margin-top: 0.75rem;")

    # ---- Upload pipelines (mirror Analysis-tab parsing) ----------------

    @render.ui
    def loc_transcript_format_status_autopilot():
        return render_transcript_format_status_ui(
            state.transcript_format_status.get(), t
        )

    @reactive.effect
    @reactive.event(input.autopilot_transcript)
    def _process_autopilot_transcript():
        file = input.autopilot_transcript()
        if file is not None:
            state.transcript_data.set(import_file(file[0]))
            detected_teacher = detect_teacher_label(file[0])
            if detected_teacher:
                ui.update_text("autopilot_name_teacher", value=detected_teacher)
                # Mirror to the Analysis-tab field so both stay in sync.
                ui.update_text("name_teacher", value=detected_teacher)
                teacher = detected_teacher
            else:
                try:
                    teacher = input.autopilot_name_teacher()
                except Exception:
                    teacher = None
            state.transcript_format_status.set(
                detect_transcript_format_status(file[0], teacher)
            )
        else:
            state.transcript_format_status.set(None)

    @reactive.effect
    @reactive.event(input.autopilot_codebook)
    def _process_autopilot_codebook():
        file = input.autopilot_codebook()
        if file is not None:
            state.codebook_data.set(import_file(file[0]))

    # ---- Provider→model dropdown sync ---------------------------------

    @reactive.effect
    def _sync_models_a():
        try:
            provider = input.autopilot_provider_a()
        except Exception:
            return
        if not provider:
            return
        models = config.get_models(provider=provider) or {}
        ui.update_select("autopilot_model_a", choices=models)

    @reactive.effect
    def _sync_models_b():
        try:
            provider = input.autopilot_provider_b()
        except Exception:
            return
        if not provider:
            return
        models = config.get_models(provider=provider) or {}
        ui.update_select("autopilot_model_b", choices=models)

    async def _run_autopilot(*, model_a: str, provider_a: str,
                             model_b: str, provider_b: str,
                             multi_coding: bool, speaker_mode: str):
        teacher_on, students_on = _speaker_flags_from_mode(speaker_mode)

        async with reactive.lock():
            autopilot_running.set(True)
            autopilot_error.set(None)
            autopilot_phase.set("coder_a_running")
            autopilot_results.set({})
            await reactive.flush()

        # Quantitative stats: same transcript for both coders, so compute once
        # before Coder A. Populates state.stats etc., which the pickle saver
        # reads later — without this, reloading an autopilot session from the
        # history crashes the Results-tab quantitative panel.
        try:
            num_class = int(input.autopilot_num_pupils() or 0)
        except Exception:
            num_class = 0
        try:
            await _compute_quant_stats(
                state,
                teacher_name=_autopilot_teacher_name(state),
                num_class=num_class,
            )
        except Exception as exc:
            print(f"[autopilot] quant-stats failed: {exc}")

        # --- Coder A ---
        df_a, raw_a, err_a = await _do_coding(
            state,
            provider=provider_a, model=model_a,
            multi_coding=multi_coding,
            teacher_on=teacher_on, students_on=students_on,
        )
        if err_a:
            async with reactive.lock():
                autopilot_phase.set("coder_a_failed")
                autopilot_error.set(err_a)
                autopilot_running.set(False)
                await reactive.flush()
            return

        _save_autopilot_pickle(state, df_a, model_a, suffix="coderA")
        cache = dict(autopilot_results.get())
        cache["a"] = {"df": df_a, "raw": raw_a, "model": model_a, "provider": provider_a}

        async with reactive.lock():
            autopilot_results.set(cache)
            autopilot_phase.set("coder_b_running")
            await reactive.flush()

        # --- Coder B ---
        df_b, raw_b, err_b = await _do_coding(
            state,
            provider=provider_b, model=model_b,
            multi_coding=multi_coding,
            teacher_on=teacher_on, students_on=students_on,
        )
        if err_b:
            async with reactive.lock():
                autopilot_phase.set("coder_b_failed")
                autopilot_error.set(err_b)
                autopilot_running.set(False)
                await reactive.flush()
            return

        _save_autopilot_pickle(state, df_b, model_b, suffix="coderB")
        cache = dict(autopilot_results.get())
        cache["b"] = {"df": df_b, "raw": raw_b, "model": model_b, "provider": provider_b}

        async with reactive.lock():
            autopilot_results.set(cache)
            report_a_df.set(_to_report_df(df_a))
            report_b_df.set(_to_report_df(df_b))
            report_a_error.set(None)
            report_b_error.set(None)
            autopilot_phase.set("done")
            autopilot_running.set(False)
            # Mark Testing tab as freshly populated. The navset update below
            # immediately switches the user to Testing, which the read-on-visit
            # effect then flips to "read" — leaving a green "data here" dot.
            state.tab_badge_testing.set("unread")
            # Autopilot füllt auch die quantitativen Stats und Coder-A-Daten,
            # die im Results-Tab sichtbar sind. Deshalb auch dort einen
            # "ungelesen"-Punkt setzen (bleibt rot, bis der User reinschaut).
            state.tab_badge_results.set("unread")
            ui.update_navset(
                "main_tabs",
                selected='<span class="shiny-html-output" id="loc_title_testing"></span>',
            )
            await reactive.flush()

    async def _run_autopilot_b_only(*, model_b: str, provider_b: str):
        cached_a = (autopilot_results.get() or {}).get("a")
        if not cached_a:
            return
        # speaker/multi-coding settings sit on the form; read them defensively
        try:
            multi_coding = bool(input.autopilot_multi_coding())
        except Exception:
            multi_coding = False
        try:
            speaker_mode = input.autopilot_speaker_mode() or "both"
        except Exception:
            speaker_mode = "both"
        teacher_on, students_on = _speaker_flags_from_mode(speaker_mode)

        async with reactive.lock():
            autopilot_running.set(True)
            autopilot_error.set(None)
            autopilot_phase.set("coder_b_running")
            await reactive.flush()

        df_b, raw_b, err_b = await _do_coding(
            state,
            provider=provider_b, model=model_b,
            multi_coding=multi_coding,
            teacher_on=teacher_on, students_on=students_on,
        )
        if err_b:
            async with reactive.lock():
                autopilot_phase.set("coder_b_failed")
                autopilot_error.set(err_b)
                autopilot_running.set(False)
                await reactive.flush()
            return

        _save_autopilot_pickle(state, df_b, model_b, suffix="coderB")
        cache = dict(autopilot_results.get())
        cache["b"] = {"df": df_b, "raw": raw_b, "model": model_b, "provider": provider_b}

        async with reactive.lock():
            autopilot_results.set(cache)
            report_a_df.set(_to_report_df(cached_a["df"]))
            report_b_df.set(_to_report_df(df_b))
            report_a_error.set(None)
            report_b_error.set(None)
            autopilot_phase.set("done")
            autopilot_running.set(False)
            state.tab_badge_testing.set("unread")
            state.tab_badge_results.set("unread")
            ui.update_navset(
                "main_tabs",
                selected='<span class="shiny-html-output" id="loc_title_testing"></span>',
            )
            await reactive.flush()

    @reactive.effect
    @reactive.event(input.autopilot_start)
    def _kickoff():
        if autopilot_running.get():
            return
        # Validate inputs.
        if state.transcript_data.get() is None or state.codebook_data.get() is None:
            autopilot_error.set(t("autopilot", "error_inputs_missing"))
            return
        try:
            provider_a = input.autopilot_provider_a()
            model_a = input.autopilot_model_a()
            provider_b = input.autopilot_provider_b()
            model_b = input.autopilot_model_b()
            multi_coding = bool(input.autopilot_multi_coding())
            speaker_mode = input.autopilot_speaker_mode() or "both"
        except Exception as exc:
            autopilot_error.set(f"Form read error: {exc}")
            return

        if not (provider_a and model_a and provider_b and model_b):
            autopilot_error.set(t("autopilot", "error_select_models"))
            return
        if (provider_a, model_a) == (provider_b, model_b):
            autopilot_error.set(t("autopilot", "error_same_model"))
            return
        if speaker_mode not in {"teacher", "students", "both"}:
            speaker_mode = "both"

        # Mirror the manual flow's pre-flight check: if teacher coding is
        # requested but the configured teacher name doesn't appear in the
        # transcript, the LLM will silently skip teacher utterances. Block
        # here with a clear message instead of letting that happen.
        teacher_on, _ = _speaker_flags_from_mode(speaker_mode)
        if teacher_on:
            teacher_name = _autopilot_teacher_name(state)
            if not _teacher_in_transcript(state.transcript_data.get(), teacher_name):
                autopilot_error.set(t("analysis", "teacher_not_found"))
                return

        autopilot_error.set(None)
        asyncio.create_task(_run_autopilot(
            model_a=model_a, provider_a=provider_a,
            model_b=model_b, provider_b=provider_b,
            multi_coding=multi_coding, speaker_mode=speaker_mode,
        ))

    @reactive.effect
    @reactive.event(input.autopilot_retry_b)
    def _kickoff_retry_b():
        if autopilot_running.get():
            return
        try:
            provider_b = input.autopilot_provider_b()
            model_b = input.autopilot_model_b()
        except Exception as exc:
            autopilot_error.set(f"Form read error: {exc}")
            return
        if not (provider_b and model_b):
            autopilot_error.set(t("autopilot", "error_select_models"))
            return
        autopilot_error.set(None)
        asyncio.create_task(_run_autopilot_b_only(
            model_b=model_b, provider_b=provider_b,
        ))
