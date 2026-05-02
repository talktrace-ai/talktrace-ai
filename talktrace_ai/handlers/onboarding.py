"""Onboarding section: welcome modal, demo loader, language toggle, startup config."""
from ._common import *


def register(state):
    input = state.input
    output = state.output
    session = state.session
    config = state.config
    t = state.t
    transcript_data = state.transcript_data
    codebook_data = state.codebook_data
    api_key_groq = state.api_key_groq
    api_key_openai = state.api_key_openai
    api_key_anthropic = state.api_key_anthropic
    api_key_ollama = state.api_key_ollama
    llm_analysis_data = state.llm_analysis_data
    model = state.model
    analysis_state = state.analysis_state
    analysis_llm_state = state.analysis_llm_state
    current_lang = state.current_lang
    code_legend_storage = state.code_legend_storage
    system_prompt = state.system_prompt
    user_prompt = state.user_prompt

    @render.ui
    def tt_demo_button_top():
        return ui.div(
            ui.input_action_button(
                "tt_demo_load_btn",
                t("onboarding", "demo_button"),
                icon=icon_svg("vial"),
                class_="btn-primary btn-sm",
            ),
            style="position: fixed; top: 0.5rem; right: 16.5rem; z-index: 1050;",
        )

    @reactive.effect
    @reactive.event(input.tt_demo_load_btn, ignore_init=True)
    async def _load_demo_from_card():
        await _load_demo_session()

    @reactive.effect
    @reactive.event(input.tt_demo_open_from_modal, ignore_init=True)
    async def _load_demo_from_modal():
        ui.modal_remove()
        await _load_demo_session()

    async def _load_demo_session():
        with reactive.isolate():
            lang = current_lang.get() if current_lang.get() in ("de", "en") else "de"
        transcript = DEMO_TRANSCRIPT[lang]
        teacher_name = DEMO_TEACHER_NAME[lang]

        # Quantitative Stats inline berechnen — der Demo-Button soll nur
        # Beispieldaten laden, nicht die Analyse-Pipeline durchlaufen
        # (kein Progress-Overlay, kein Streaming-Pfad).
        stats_df = dialog_stats(transcript, teacher_name)
        sps_df = dialog_stats_per_speaker(transcript, teacher_name)
        n_part = count_pupils(transcript)
        part_rate = (n_part / DEMO_NUM_PUPILS * 100) if DEMO_NUM_PUPILS else 0

        def _safe(speaker, col, default=0):
            m = stats_df.loc[stats_df['Sprecher'] == speaker, col]
            return m.values[0] if not m.empty else default

        # Gleiches Muster wie _restore_session_state in _session.py:
        # synchrone Sets in isolate, danach update_navset außerhalb.
        with reactive.isolate():
            transcript_data.set(transcript)
            codebook_data.set(DEMO_CODEBOOK[lang])
            llm_analysis_data.set([build_demo_llm_analysis_df(lang)])
            analysis_llm_state.set(True)
            code_legend_storage.set(DEMO_CODE_LEGEND[lang])
            ui.update_text("name_group", value=DEMO_GROUP_ID[lang])
            ui.update_numeric("num_pupils", value=DEMO_NUM_PUPILS)
            ui.update_text("name_teacher", value=teacher_name)
            ui.update_switch("llm_switch", value=False)

            state.num_participants.set(n_part)
            state.stats.set(stats_df)
            state.stats_per_speaker.set(sps_df)
            state.teacher_impulses_count.set(count_teacher_impulses(stats_df, teacher_name))
            state.participation_rate.set(part_rate)
            state.t_turns.set(_safe(teacher_name, 'Anzahl_Beitraege'))
            state.t_turns_length.set(round(_safe(teacher_name, 'Durchschnitt_Woerter'), 1))
            state.t_turns_length_mean_sd.set(round(_safe(teacher_name, 'Median_Woerter'), 1))
            state.p_turns.set(_safe("Schüler:innen", 'Anzahl_Beitraege'))
            state.p_turns_length.set(round(_safe("Schüler:innen", 'Durchschnitt_Woerter'), 1))
            state.p_turns_length_mean_sd.set(_safe("Schüler:innen", 'Median_Woerter'))

            analysis_state.set(True)
            # Demo loader switches to Results immediately — no "unread"
            # alert needed; mark the tab as data-present-and-seen.
            state.tab_badge_results.set("read")

        ui.update_navset("main_tabs", selected='<span class="shiny-html-output" id="loc_title_results"></span>')
        ui.notification_show(t("onboarding", "demo_loaded"), type="message", duration=4)

    @render.ui
    def tt_quickstart_panel():
        # Aktuell ausgewählten Anbieter berücksichtigen (re-rendert bei Wechsel)
        try:
            provider = input.provider_select()
        except Exception:
            provider = config.get_current_api()

        api_keys = {
            "groq": api_key_groq.get(),
            "openai": api_key_openai.get(),
            "anthropic": api_key_anthropic.get(),
            "ollama": api_key_ollama.get(),
        }
        has_key_for_provider = bool(api_keys.get(provider))

        try:
            llm_on = bool(input.llm_switch())
        except Exception:
            llm_on = True

        items = [
            (t("onboarding", "quickstart_api_key"), bool(has_key_for_provider) or not llm_on),
            (t("onboarding", "quickstart_model"), bool(model.get()) or not llm_on),
            (t("onboarding", "quickstart_transcript"), transcript_data.get() is not None),
        ]
        if llm_on:
            items.append((t("onboarding", "quickstart_codebook"), codebook_data.get() is not None))
        items.append((t("onboarding", "quickstart_analysis_done"), bool(analysis_state.get())))

        all_ok = all(ok for _, ok in items)
        status_label = t("onboarding", "quickstart_status_ok") if all_ok else t("onboarding", "quickstart_status_pending")
        return ui.tags.div(
            ui.tags.div(
                ui.tags.span(t("onboarding", "quickstart_title") + " — " + status_label),
                ui.tags.span("▾", class_="qs-caret"),
                class_="qs-header",
            ),
            ui.tags.div(
                *[ui.tags.div(
                    ui.tags.span("✓" if ok else "✗", class_=f"qs-icon {'ok' if ok else 'pending'}"),
                    ui.tags.span(label),
                    class_="qs-item",
                ) for label, ok in items],
                class_="qs-body",
            ),
            id="tt-quickstart",
            class_=f"qs-{'ok' if all_ok else 'pending'}",
        )

    def _make_welcome_modal():
        return ui.modal(
            ui.p(t("onboarding", "welcome_intro")),
            ui.tags.ol(
                ui.tags.li(t("onboarding", "welcome_step_1")),
                ui.tags.li(t("onboarding", "welcome_step_2")),
                ui.tags.li(t("onboarding", "welcome_step_3")),
            ),
            ui.tags.hr(),
            ui.p(t("onboarding", "welcome_demo_hint"), class_="text-muted"),
            ui.input_action_button(
                "tt_demo_open_from_welcome",
                t("onboarding", "demo_button"),
                icon=icon_svg("vial"),
                class_="btn-primary btn-sm",
            ),
            title=t("onboarding", "welcome_title"),
            easy_close=True,
            footer=ui.modal_button(t("onboarding", "welcome_close"), class_="btn-success"),
            size="m",
        )

    @reactive.effect
    @reactive.event(input.tt_demo_open_from_welcome, ignore_init=True)
    async def _load_demo_from_welcome():
        ui.modal_remove()
        await _load_demo_session()

    # Data-protection acknowledgment: hard gate before the welcome modal.
    # Shown on first launch (and again after a manual reset of the flag
    # file) so the user has to actively pick which kind of data they will
    # work with — explicit-consent real data, or fictive test data.
    # easy_close=False + no dismiss button means the only path forward is
    # the "I acknowledge" button, which is disabled until a radio is set.
    def _make_dataprotection_modal():
        return ui.modal(
            ui.tags.div(
                icon_svg("shield-halved"),
                " ",
                ui.tags.strong(t("onboarding", "dp_intro_strong")),
                style="font-size:1.05rem;margin-bottom:0.5rem;",
            ),
            ui.p(t("onboarding", "dp_intro_body")),
            ui.input_radio_buttons(
                "dp_data_kind",
                t("onboarding", "dp_choice_label"),
                choices={
                    "consent": t("onboarding", "dp_choice_consent"),
                    "fictive": t("onboarding", "dp_choice_fictive"),
                },
                selected=None,
            ),
            ui.tags.p(
                t("onboarding", "dp_consent_hint"),
                class_="text-muted small",
            ),
            title=t("onboarding", "dp_title"),
            easy_close=False,
            footer=ui.input_action_button(
                "tt_dataprotection_confirm",
                t("onboarding", "dp_confirm"),
                icon=icon_svg("check"),
                class_="btn-success",
            ),
            size="m",
        )

    @reactive.effect
    @reactive.event(input.tt_dataprotection_confirm, ignore_init=True)
    def _confirm_dataprotection():
        # Force the user to pick one of the two options before moving on.
        try:
            choice = input.dp_data_kind()
        except Exception:
            choice = None
        if choice not in ("consent", "fictive"):
            ui.notification_show(
                t("onboarding", "dp_pick_required"),
                type="warning",
                duration=4,
            )
            return
        _mark_dataprotection_acknowledged()
        ui.modal_remove()
        # If this was the first launch, immediately follow with the welcome
        # modal so onboarding stays a single coherent flow.
        if not _welcome_shown():
            ui.modal_show(_make_welcome_modal())
            _mark_welcome_shown()

    def _maybe_show_welcome():
        # Order: data-protection first (blocking), then welcome. Both have
        # their own once-per-install flag file under config/.
        if not _dataprotection_acknowledged():
            with reactive.isolate():
                ui.modal_show(_make_dataprotection_modal())
            return
        if _welcome_shown():
            return
        with reactive.isolate():
            ui.modal_show(_make_welcome_modal())
        _mark_welcome_shown()

    session.on_flushed(_maybe_show_welcome, once=True)

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


    # Fill Fields from Config File
    ui.update_text("name_group", value=config.get_parameters()['group_id'])
    ui.update_numeric("num_pupils", value=config.get_parameters()['num_pupils'])
    ui.update_text("name_teacher", value=config.get_parameters()['teacher_name'])
    ui.update_text("name_teacher_options", value=config.get_parameters()['teacher_name'])
    ui.update_text("name_group_options", value=config.get_parameters()['group_id'])
    ui.update_numeric("num_pupils_options", value=config.get_parameters()['num_pupils'])
    ui.update_action_button("language_toggle", icon=icon_svg("globe"), label="English" if config.get_localization()['current_language'] == 'de' else "Deutsch")


    api_key_openai.set(safe_get_password("talktrace", "api_key_openai"))
    api_key_groq.set(safe_get_password("talktrace", "api_key_groq"))
    api_key_anthropic.set(safe_get_password("talktrace", "api_key_anthropic"))
    api_key_ollama.set(safe_get_password("talktrace", "api_key_ollama"))
