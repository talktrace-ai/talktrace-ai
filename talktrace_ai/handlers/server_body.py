"""Combined server-body register function.

Hosts every reactive handler that previously lived inline inside
app.py's server(). Kept as a single function so cross-section
closures (e.g. onboarding's demo loader calling sidebar's
run_analysis) continue to resolve via lexical scope rather than
needing to be plumbed through AppState.
"""
from ._common import *


def register(state):
    input = state.input
    output = state.output
    session = state.session
    config = state.config
    t = state.t
    transcript_data = state.transcript_data
    codebook_data = state.codebook_data
    converted_transcript = state.converted_transcript
    fmt_text = state.fmt_text
    fmt_analysis = state.fmt_analysis
    fmt_options = state.fmt_options
    fmt_meta = state.fmt_meta
    api_key_groq = state.api_key_groq
    api_key_openai = state.api_key_openai
    api_key_anthropic = state.api_key_anthropic
    api_key_ollama = state.api_key_ollama
    ollama_status_refresh = state.ollama_status_refresh
    current_api = state.current_api
    num_participants = state.num_participants
    participation_rate = state.participation_rate
    t_turns = state.t_turns
    t_turns_length = state.t_turns_length
    t_turns_length_mean_sd = state.t_turns_length_mean_sd
    p_turns = state.p_turns
    p_turns_length = state.p_turns_length
    p_turns_length_mean_sd = state.p_turns_length_mean_sd
    stats = state.stats
    stats_per_speaker = state.stats_per_speaker
    llm_analysis_data = state.llm_analysis_data
    model = state.model
    teacher_impulses_count = state.teacher_impulses_count
    analysis_state = state.analysis_state
    analysis_llm_state = state.analysis_llm_state
    sim_plot = state.sim_plot
    qual_plot = state.qual_plot
    qual_stats_df = state.qual_stats_df
    placeholder_plot = state.placeholder_plot
    model_deleted = state.model_deleted
    current_lang = state.current_lang
    code_legend_storage = state.code_legend_storage
    estimated_cost = state.estimated_cost
    token_count = state.token_count
    report_a_df = state.report_a_df
    report_b_df = state.report_b_df
    report_a_error = state.report_a_error
    report_b_error = state.report_b_error

    # === section: onboarding ===

    ### Onboarding ----------------------------------------------------------

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
            transcript_data.set(DEMO_TRANSCRIPT)
            llm_analysis_data.set([build_demo_llm_analysis_df()])
            analysis_llm_state.set(True)
            code_legend_storage.set(DEMO_CODE_LEGEND)
            ui.update_text("name_group", value=DEMO_GROUP_ID)
            ui.update_numeric("num_pupils", value=DEMO_NUM_PUPILS)
            ui.update_text("name_teacher", value=DEMO_TEACHER_NAME)
            ui.update_switch("llm_switch", value=False)
        await run_analysis(force_no_llm=True)
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

    def _maybe_show_welcome():
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


    api_key_openai.set(safe_get_password("talktrace", "api_key_openai"))
    api_key_groq.set(safe_get_password("talktrace", "api_key_groq"))
    api_key_anthropic.set(safe_get_password("talktrace", "api_key_anthropic"))
    api_key_ollama.set(safe_get_password("talktrace", "api_key_ollama"))



    # === section: sidebar ===

    ### Sidebar --------------------------------------------------------
    # Model Selection
    @render.ui
    def loc_dynamic_model_select():
        return ui.div(
            ui.input_select("provider_select", t("sidebar", "provider_select"), choices={"openai": "OpenAI", "groq": "Groq", "anthropic": "Anthropic", "ollama": "Ollama"}, selected=config.get_current_api()),
            ui.input_select("model_select", t("sidebar", "model_select"), choices=select_api_choices(), selected=config.get_current_model()),
            **{"data-tt-help": t("onboarding", "tooltip_model_select")},
        )


    @reactive.effect()
    def update_current_provider():
        selected_provider = input.provider_select()
        if not selected_provider:
            return
        if selected_provider == config.get_current_api():
            return
        config.set_current_api(selected_provider)
        current_api.set(selected_provider)
        # passendes erstes Modell des neuen Anbieters auswählen
        available_models = select_api_choices()
        if available_models:
            first_model = next(iter(available_models))
            model.set(first_model)
            config.set_current_model(first_model)
            ui.update_select("model_select", choices=available_models, selected=first_model)


    @reactive.effect()
    def update_current_model():
        model.set(input.model_select())
        config.set_current_model(input.model_select())


    # LLM Analyse
    @render.ui
    def loc_llm_switch():
        return ui.div(
            ui.input_switch("llm_switch", t("sidebar", "llm_switch"), True),
            **{"data-tt-help": t("onboarding", "tooltip_llm_switch")},
        )


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
        teacher_name = input.name_teacher() or t("analysis", "name_teacher_var")
        if teacher and students:
            return ""
        if teacher and not students:
            return t("sidebar", f"{prefix}_teacher_only").format(teacher_name=teacher_name)
        if students and not teacher:
            return t("sidebar", f"{prefix}_students_only").format(teacher_name=teacher_name)
        return t("sidebar", f"{prefix}_none")

    def _sanitize_prompt_for_speakers(text: str, teacher: bool, students: bool) -> str:
        """Remove teacher/student references from prompt text when that group is disabled.
        Keeps the text grammatical for the default prompts; custom prompts are
        handled best-effort."""
        if teacher and students:
            return text
        # --- German references -------------------------------------------------
        if not teacher:
            text = text.replace("Lehrperson UND Schüler:innen", "Schüler:innen")
            text = text.replace("Lehrperson und Schüler:innen", "Schüler:innen")
            text = text.replace("ALLER Sprecher:innen (Lehrperson UND Schüler:innen)", "der Schüler:innen")
            text = text.replace("ALLER Sprecher:innen", "der Schüler:innen")
            text = text.replace("Lehrperson", "")
            text = text.replace("LEHRER", "")
            text = text.replace("Lehrer", "")
        if not students:
            text = text.replace("Lehrperson UND Schüler:innen", "Lehrperson")
            text = text.replace("Lehrperson und Schüler:innen", "Lehrperson")
            text = text.replace("ALLER Sprecher:innen (Lehrperson UND Schüler:innen)", "der Lehrperson")
            text = text.replace("ALLER Sprecher:innen", "der Lehrperson")
            text = text.replace("Schüler:innen", "")
            text = text.replace("S01, S02, S03…", "")
            text = text.replace("S01, S02, S03...", "")
            text = text.replace("S01, S02, S03", "")
            text = text.replace("S01", "")
        # --- English references ------------------------------------------------
        if not teacher:
            text = text.replace("teacher AND students", "students")
            text = text.replace("teacher and students", "students")
            text = text.replace("ALL speakers (teacher AND students)", "students")
            text = text.replace("ALL speakers", "students")
            text = text.replace("teacher", "")
        if not students:
            text = text.replace("teacher AND students", "teacher")
            text = text.replace("teacher and students", "teacher")
            text = text.replace("ALL speakers (teacher AND students)", "teacher")
            text = text.replace("ALL speakers", "teacher")
            text = text.replace("students", "")
            text = text.replace("S01, S02, S03…", "")
            text = text.replace("S01, S02, S03...", "")
            text = text.replace("S01, S02, S03", "")
            text = text.replace("S01", "")
        # --- Cleanup whitespace artifacts --------------------------------------
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*-\s*", "-", text)
        return text.strip()

    @reactive.calc
    def effective_system_prompt():
        teacher, students = _speaker_flags()
        base = _sanitize_prompt_for_speakers(system_prompt.get(), teacher, students)
        return base + _speaker_filter_suffix("system")

    @reactive.calc
    def effective_user_prompt():
        teacher, students = _speaker_flags()
        raw = _sanitize_prompt_for_speakers(user_prompt.get(), teacher, students)
        suffix = _speaker_filter_suffix("user")
        if not suffix:
            return raw
        # LLMs suffer from "lost in the middle" on very long contexts.
        # The speaker-filter instruction must sit RIGHT AFTER the transcript
        # block, not at the very end after thousands of tokens of codebook.
        if "{transcript}" in raw:
            target = "{transcript}"
            idx = raw.index(target)
            insert_pos = idx + len(target)
            return raw[:insert_pos] + "\n\n" + suffix + raw[insert_pos:]
        return raw + suffix


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
    async def run_analysis(force_no_llm: bool = False):
        req(transcript_data.get() != None)
        # If teacher analysis is desired, verify the name exists in the transcript.
        teacher_name = input.name_teacher()
        transcript = transcript_data.get()
        teacher_on, students_on = _speaker_flags()
        if teacher_on:
            # Simple check: exact or case-insensitive word boundary match.
            search_name = re.escape(teacher_name) if teacher_name else ""
            found = False
            if search_name:
                # Check as a standalone speaker label ("Name:" pattern) anywhere in the text.
                pattern = re.compile(rf"^\s*" + search_name + r"\s*:", re.IGNORECASE | re.MULTILINE)
                if pattern.search(transcript):
                    found = True
                # Also allow plain substring match as a fallback.
                elif teacher_name.lower() in transcript.lower():
                    found = True
            if not found:
                return t("analysis", "teacher_not_found")
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
            if input.llm_switch() and not force_no_llm:
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
        return ui.input_action_button(
            "button_report_open",
            t("sidebar", "download_report"),
            icon=icon_svg("download"),
            class_="btn-sm",
        ),


    report_options = reactive.value({
        "sections": dict(DEFAULT_REPORT_SECTIONS),
        "format": "docx",
    })


    @reactive.effect
    @reactive.event(input.button_report_open)
    def _open_report_modal():
        opts = report_options.get()
        sec = opts["sections"]
        quali_available = bool(analysis_llm_state.get()) and bool(llm_analysis_data.get())
        quali_default = sec.get("quali", True) and quali_available
        quali_ot_default = sec.get("over_time_quali", False) and quali_available
        legend_default = sec.get("legend", True) and quali_available

        sections_block = ui.div(
            ui.tags.label(t("report_options", "sections_label"), class_="form-label fw-bold"),
            ui.input_checkbox("report_sec_quant", t("report_options", "sec_quant"), value=sec.get("quant", True)),
            ui.input_checkbox("report_sec_over_time_quant", t("report_options", "sec_over_time_quant"), value=sec.get("over_time_quant", False)),
            ui.input_checkbox("report_sec_quali", t("report_options", "sec_quali"), value=quali_default),
            ui.input_checkbox("report_sec_over_time_quali", t("report_options", "sec_over_time_quali"), value=quali_ot_default),
            ui.input_checkbox("report_sec_legend", t("report_options", "sec_legend"), value=legend_default),
        )
        if not quali_available:
            sections_block = ui.div(
                sections_block,
                ui.tags.p(t("report_options", "quali_disabled_hint"), class_="text-muted small"),
            )

        format_block = ui.div(
            ui.input_radio_buttons(
                "report_format",
                t("report_options", "format_label"),
                choices={
                    "docx": t("report_options", "format_docx"),
                    "pdf": t("report_options", "format_pdf"),
                    "xlsx": t("report_options", "format_xlsx"),
                    "html": t("report_options", "format_html"),
                },
                selected=opts.get("format", "docx"),
                inline=True,
            ),
        )

        body = ui.div(
            ui.tags.p(t("report_options", "dialog_intro")),
            sections_block,
            ui.tags.hr(),
            format_block,
            ui.tags.hr(),
            ui.div(
                ui.download_button(
                    "download_report",
                    t("report_options", "download_now"),
                    icon=icon_svg("download"),
                    class_="btn-primary",
                ),
                " ",
                ui.input_action_button(
                    "button_report_cancel",
                    t("report_options", "cancel"),
                    class_="btn-secondary",
                ),
                style="display:flex;gap:0.5rem;justify-content:flex-end",
            ),
        )

        ui.modal_show(ui.modal(
            body,
            title=t("report_options", "dialog_title"),
            easy_close=True,
            footer=None,
            size="m",
        ))


    @reactive.effect
    @reactive.event(input.button_report_cancel)
    def _close_report_modal():
        ui.modal_remove()


    def _current_report_sections():
        try:
            sec = {
                "quant": bool(input.report_sec_quant()),
                "over_time_quant": bool(input.report_sec_over_time_quant()),
                "quali": bool(input.report_sec_quali()),
                "over_time_quali": bool(input.report_sec_over_time_quali()),
                "legend": bool(input.report_sec_legend()),
            }
        except Exception:
            sec = dict(DEFAULT_REPORT_SECTIONS)
        return sec


    def _current_report_format():
        try:
            return input.report_format() or "docx"
        except Exception:
            return "docx"


    @render.download(filename=lambda: f"{date.today().isoformat()} - TalkTrace AI {t('results', 'results_group')} {input.name_group()}.{_current_report_format()}")
    def download_report():
        sections = _current_report_sections()
        fmt = _current_report_format()
        # Persist last selection for the next modal open.
        report_options.set({"sections": dict(sections), "format": fmt})

        if not any(sections.values()):
            ui.notification_show(t("report_options", "no_section_selected"), type="warning", duration=4)
            return None

        suffix = f".{fmt}"
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.close()

        has_llm = bool(llm_analysis_data.get())
        impulse_table = qual_stats_df.get() if has_llm else None
        plot_qual = qual_plot.get() if has_llm else None

        plot_ot_quant = None
        plot_ot_quali = None
        df_ot_quant = None
        df_ot_quali = None
        if sections.get("over_time_quant") and transcript_data.get() is not None:
            try:
                plot_ot_quant = make_sim_stats_over_time_plot()
                teacher_name = input.name_teacher() or t("analysis", "name_teacher_var")
                df_ot_quant = dialog_stats_over_time(
                    transcript_data.get(), teacher_name,
                    n_segments=3, segment_labels=_segment_labels_for(3),
                )
            except Exception as e:
                print(f"[REPORT] over-time quant plot failed: {e}")
        if sections.get("over_time_quali") and has_llm and transcript_data.get() is not None:
            try:
                plot_ot_quali = make_qualitative_stats_over_time_plot()
                latest_df = llm_analysis_data.get()[-1] if llm_analysis_data.get() else None
                if latest_df is not None:
                    teacher_name = input.name_teacher() or t("analysis", "name_teacher_var")
                    mapped = map_impulses_to_turn_index(latest_df, transcript_data.get(), teacher_name)
                    total_turns = count_transcript_turns(transcript_data.get(), teacher_name)
                    df_ot_quali = code_distribution_over_time(
                        mapped, total_turns, n_segments=3, segment_labels=_segment_labels_for(3),
                    )
            except Exception as e:
                print(f"[REPORT] over-time quali plot failed: {e}")

        try:
            generate_report2(
                tmp_file.name,
                input.name_group(), input.num_pupils(), num_participants.get(), participation_rate.get(),
                {"num": t_turns.get(), "words": t_turns_length.get(), "mean_sd": t_turns_length_mean_sd.get()},
                {"num": p_turns.get(), "words": p_turns_length.get(), "mean_sd": p_turns_length_mean_sd.get()},
                sim_plot.get(),
                teacher_impulses_count.get(),
                caption=code_legend_storage.get(),
                plot_impulse_coding=plot_qual,
                impulse_table=impulse_table,
                plot_distribution_over_time=plot_ot_quant,
                plot_coding_over_time=plot_ot_quali,
                dist_over_time_df=df_ot_quant,
                code_over_time_df=df_ot_quali,
                sections=sections,
                output_format=fmt,
                model_name=model.get() or "",
            )
        except RuntimeError as e:
            key = str(e)
            msg = t("report_options", key) if key in ("pdf_unavailable", "pdf_unavailable_linux", "xlsx_unavailable") else str(e)
            ui.notification_show(msg, type="error", duration=6)
            return None

        ui.modal_remove()
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
        return ui.download_button("button_export_session", t("sidebar", "export_session"), icon = icon_svg("file-export"), class_="btn-sm"),

    
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


    # Verlauf (History) — manueller Save + Liste der letzten Sitzungen
    history_version = reactive.value(0)


    @render.ui
    def loc_button_history():
        return ui.input_action_button(
            "button_history",
            t("sidebar", "history_button"),
            icon=icon_svg("clock-rotate-left"),
            class_="btn-sm",
        ),


    def _history_format_row(entry):
        date_str = entry.get("saved_at", "").replace("T", " ")[:16]
        group = entry.get("group_id", "")
        model_name = entry.get("model", "")
        n_turns = entry.get("n_turns", 0)
        return f"{date_str} · {group or '—'} · {model_name or '—'} · {n_turns} {t('sidebar', 'history_col_turns')}"


    def _show_history_modal():
        # Re-read fresh entries every time the modal is shown.
        history_version.get()  # establish reactive dep so re-renders re-show
        entries = list_history()
        if entries:
            choices = {e["filename"]: _history_format_row(e) for e in entries}
            picker = ui.input_select(
                "history_select",
                t("sidebar", "history_select_label"),
                choices=choices,
            )
            actions = ui.div(
                ui.input_action_button(
                    "history_load_btn",
                    t("sidebar", "history_load"),
                    icon=icon_svg("file-arrow-up"),
                    class_="btn-success",
                ),
                " ",
                ui.input_action_button(
                    "history_delete_btn",
                    t("sidebar", "history_delete"),
                    icon=icon_svg("trash"),
                    class_="btn-danger",
                ),
                style="margin-top: 0.5rem;",
            )
        else:
            picker = ui.p(t("sidebar", "history_empty"))
            actions = None

        body = ui.div(
            ui.input_action_button(
                "history_save_btn",
                t("sidebar", "history_save_now"),
                icon=icon_svg("floppy-disk"),
                class_="btn-primary",
            ),
            ui.tags.hr(),
            picker,
            actions,
        )
        ui.modal_show(ui.modal(
            body,
            title=t("sidebar", "history_title"),
            easy_close=True,
            footer=ui.modal_button(t("sidebar", "history_close"), class_="btn-default"),
            size="l",
        ))


    @reactive.effect
    @reactive.event(input.button_history)
    def open_history_modal():
        _show_history_modal()


    @reactive.effect
    @reactive.event(input.history_save_btn)
    def save_current_to_history():
        if not analysis_state.get() or stats.get() is None:
            ui.modal_remove()
            ui.modal_show(ui.modal(
                t("sidebar", "history_save_blocked"),
                title=t("analysis", "modal_title_attention"),
                easy_close=True,
                footer=ui.modal_button("OK", class_="btn-success"),
            ))
            return
        session_data = {
            "transcript_data": transcript_data.get(),
            "num_participants": num_participants.get(),
            "participation_rate": participation_rate.get(),
            "stats": stats.get(),
            "llm_analysis_data": llm_analysis_data.get(),
            "analysis_llm_state": analysis_llm_state.get(),
            "code_legend_storage": code_legend_storage.get(),
        }
        try:
            n_turns = int(stats.get()['Anzahl_Beitraege'].sum()) if stats.get() is not None else 0
        except Exception:
            n_turns = 0
        save_to_history(
            session_data,
            group_id=input.name_group() or "",
            model=config.get_current_model() or "",
            n_turns=n_turns,
            n_pupils=num_participants.get(),
            participation_rate=participation_rate.get(),
            language=config.get_localization().get("current_language"),
        )
        history_version.set(history_version.get() + 1)
        ui.modal_remove()
        _show_history_modal()


    @reactive.effect
    @reactive.event(input.history_delete_btn)
    def delete_history_selected():
        fname = input.history_select()
        if not fname:
            return
        delete_history_entry(fname)
        history_version.set(history_version.get() + 1)
        ui.modal_remove()
        _show_history_modal()


    @reactive.effect
    @reactive.event(input.history_load_btn)
    async def load_history_selected():
        fname = input.history_select()
        if not fname:
            return
        try:
            session_data = load_history_entry(fname)
        except (OSError, pickle.UnpicklingError):
            return
        with reactive.isolate():
            try:
                transcript_data.set(session_data.get("transcript_data"))
                num_participants.set(session_data.get("num_participants"))
                participation_rate.set(session_data.get("participation_rate"))
                stats.set(session_data.get("stats"))
                llm_analysis_data.set(session_data.get("llm_analysis_data"))
                analysis_llm_state.set(session_data.get("analysis_llm_state"))
                code_legend_storage.set(session_data.get("code_legend_storage"))
                ui.update_switch("llm_switch", value=False)
            except Exception:
                pass
        ui.modal_remove()
        await run_analysis()


    # Reset Session
    @render.ui
    def loc_button_reset():
        return ui.input_action_button("button_reset", t("sidebar", "reset_session"), icon = icon_svg("arrow-rotate-left"), class_="btn-danger btn-sm"),

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


    # === section: analysis ===

    ### Analyse --------------------------------------------------------

    @render.text
    def loc_title_analysis():
        return (t("analysis", "tab_title"))

    # Allgemeine Informationen
    @render.ui
    def loc_general_info():
        return ui.p(t("analysis", "general_info"))
    
    def _tt_wrap(child, key):
        return ui.div(child, **{"data-tt-help": t("onboarding", key)})

    @render.ui
    def loc_group_id():
        return _tt_wrap(
            ui.input_text("name_group", t("analysis", "group_id"), "B1"),
            "tooltip_group_id",
        )

    @render.ui
    def loc_num_pupils():
        return _tt_wrap(
            ui.input_numeric("num_pupils", t("analysis", "num_pupils"), 25, min=1, max=100),
            "tooltip_num_pupils",
        )

    @render.ui
    def loc_name_teacher():
        return _tt_wrap(
            ui.input_text("name_teacher", t("analysis", "name_teacher"), config.get_parameters()['teacher_name']),
            "tooltip_name_teacher",
        )
    
    # Dokumenteneingabe
    @render.ui
    def loc_document_input():
        return ui.p(t("analysis", "document_input"))

    # Transkript Upload
    @render.ui
    def loc_upload_transcript():
        return ui.div(
            ui.div(
                ui.tags.label(
                    t("analysis", "upload_transcript"),
                    class_="control-label",
                    style="display: block; margin-bottom: 0.25rem;",
                ),
                ui.input_file(
                    "transcript",
                    None,
                    multiple=False,
                    accept=[".txt", ".docx", ".pdf"],
                    button_label=t("analysis", "browse"),
                    placeholder=t("analysis", "placeholder"),
                ),
                class_="ttai-file-wrap",
                style="flex: 1 1 auto; min-width: 0;",
                **{"data-tt-help": t("onboarding", "tooltip_upload_transcript")},
            ),
            ui.div(
                ui.tags.label(
                    t("analysis", "check_format"),
                    class_="control-label",
                    style="display: block; margin-bottom: 0.25rem;",
                ),
                ui.tooltip(
                    ui.input_action_button(
                        "button_check_format",
                        "",
                        icon=icon_svg("wand-magic-sparkles"),
                        class_="btn-default btn-file",
                    ),
                    t("analysis", "check_format_tooltip"),
                    placement="right",
                ),
                style="flex: 0 0 auto;",
            ),
            ui.tags.style(
                ".ttai-file-wrap .shiny-input-container,"
                ".ttai-file-wrap .form-group { margin-bottom: 0 !important; }"
                ".ttai-file-wrap .control-label:empty { display: none !important; }"
            ),
            style="display: flex; gap: 0.5rem; align-items: start;",
        )

    # Transkript verarbeiten
    @reactive.effect
    @reactive.event(input.transcript)
    def process_transcript():
        file = input.transcript()
        if file is not None:
            data = import_file(file[0])
            transcript_data.set(data)
            if isinstance(data, str):
                n = count_pupils(data)
                if n > 0:
                    ui.update_numeric("num_pupils", value=n)

    # Transkript-Format prüfen und ggf. konvertieren (mehrstufiger Wizard)
    def _bracket_id(delim: str) -> str:
        return {
            "[]": "sq",
            "()": "rd",
            "{}": "cu",
            "<>": "an",
            "//": "sl",
            "**": "st",
        }.get(delim, "x")

    def _build_speaker_options(n_speakers: int) -> dict[str, str]:
        opts = {"TEACHER": t("analysis", "format_speaker_role_teacher")}
        for i in range(1, n_speakers + 1):
            key = f"S{i:02d}"
            opts[key] = t("analysis", "format_speaker_role_student_n").format(n=i)
        opts["__ignore__"] = t("analysis", "format_speaker_role_ignore")
        return opts

    def _show_stage_speakers():
        analysis = fmt_analysis.get()
        options = fmt_options.get()
        if analysis is None or options is None:
            return
        if not analysis.speakers:
            ui.modal_show(ui.modal(
                t("analysis", "format_modal_no_speakers"),
                title=t("analysis", "modal_title_format_check"),
                easy_close=True,
                footer=ui.modal_button(t("analysis", "modal_button_close"), class_="btn-success"),
            ))
            return
        select_opts = _build_speaker_options(len(analysis.speakers))
        rows = []
        for i, raw in enumerate(analysis.speakers):
            mapped = options.speaker_map.get(raw)
            selected = mapped if mapped is not None else "__ignore__"
            if selected not in select_opts:
                selected = "__ignore__"
            rows.append(ui.tags.tr(
                ui.tags.td(raw, style="padding: 0.25rem 0.5rem; font-family: monospace;"),
                ui.tags.td(
                    ui.input_select(
                        f"fmt_spk_{i}", None, choices=select_opts, selected=selected,
                    ),
                    style="padding: 0.25rem 0.5rem;",
                ),
            ))
        table = ui.tags.table(
            ui.tags.thead(ui.tags.tr(
                ui.tags.th(t("analysis", "format_modal_speakers_col_raw")),
                ui.tags.th(t("analysis", "format_modal_speakers_col_target")),
            )),
            ui.tags.tbody(*rows),
            class_="table table-sm",
            style="width: 100%;",
        )
        ui.modal_show(ui.modal(
            ui.p(t("analysis", "format_modal_speakers_intro")),
            table,
            title=t("analysis", "format_modal_speakers_title"),
            easy_close=False,
            size="l",
            footer=ui.tags.div(
                ui.modal_button(t("analysis", "modal_button_cancel"), class_="btn-secondary"),
                ui.input_action_button(
                    "button_fmt_to_brackets",
                    t("analysis", "format_modal_button_next"),
                    class_="btn-success",
                ),
            ),
        ))

    def _show_stage_brackets():
        analysis = fmt_analysis.get()
        options = fmt_options.get()
        if analysis is None or options is None:
            return
        if not analysis.bracket_patterns and not analysis.other_tokens:
            _show_stage_preview()
            return
        rows = []
        for grp in analysis.bracket_patterns:
            bid = _bracket_id(grp.delimiter)
            samples = ", ".join(grp.samples) if grp.samples else ""
            current = "strip" if options.strip_brackets.get(grp.delimiter) else "keep"
            rows.append(ui.div(
                ui.tags.b(f"{grp.delimiter} ({grp.count}×)"),
                ui.tags.span(
                    f"  {t('analysis', 'format_bracket_samples')}: {samples}",
                    style="color: #888; margin-left: 0.5rem;",
                ),
                ui.input_radio_buttons(
                    f"fmt_br_{bid}", None,
                    choices={
                        "keep": t("analysis", "format_bracket_keep"),
                        "strip": t("analysis", "format_bracket_strip"),
                    },
                    selected=current,
                    inline=True,
                ),
                style="margin-bottom: 0.75rem; padding: 0.5rem; border-bottom: 1px solid #444;",
            ))
        for i, grp in enumerate(analysis.other_tokens):
            current = "strip" if options.strip_tokens.get(grp.token, True) else "keep"
            rows.append(ui.div(
                ui.tags.b(f"{grp.token!r} ({grp.count}×)"),
                ui.input_radio_buttons(
                    f"fmt_tok_{i}", None,
                    choices={
                        "keep": t("analysis", "format_bracket_keep"),
                        "strip": t("analysis", "format_bracket_strip"),
                    },
                    selected=current,
                    inline=True,
                ),
                style="margin-bottom: 0.75rem; padding: 0.5rem; border-bottom: 1px solid #444;",
            ))
        ui.modal_show(ui.modal(
            ui.p(t("analysis", "format_modal_brackets_intro")),
            *rows,
            title=t("analysis", "format_modal_brackets_title"),
            easy_close=False,
            size="l",
            footer=ui.tags.div(
                ui.input_action_button(
                    "button_fmt_back_speakers",
                    t("analysis", "format_modal_button_back"),
                    class_="btn-secondary",
                ),
                ui.input_action_button(
                    "button_fmt_to_preview",
                    t("analysis", "format_modal_button_convert"),
                    class_="btn-success",
                ),
            ),
        ))

    def _show_stage_preview():
        text = fmt_text.get()
        options = fmt_options.get()
        meta = fmt_meta.get()
        if text is None or options is None or meta is None:
            return
        converted = convert_with_options(text, options)
        base = os.path.splitext(meta["name"])[0]
        ext = meta["ext"]
        out_ext = ".txt" if ext == ".txt" else ".docx"
        converted_transcript.set({
            "text": converted,
            "ext": out_ext,
            "filename": f"{base}_converted{out_ext}",
        })
        preview = "\n".join(converted.splitlines()[:10])
        ui.modal_show(ui.modal(
            ui.p(t("analysis", "modal_format_invalid_confirm")),
            ui.tags.pre(preview, style="max-height: 300px; overflow: auto;"),
            title=t("analysis", "format_modal_preview_title"),
            easy_close=False,
            size="l",
            footer=ui.tags.div(
                ui.input_action_button(
                    "button_fmt_back_brackets",
                    t("analysis", "format_modal_button_back"),
                    class_="btn-secondary",
                ),
                ui.download_button(
                    "download_converted_transcript",
                    t("analysis", "download_converted"),
                    icon=icon_svg("download"),
                    class_="btn-success",
                ),
            ),
        ))

    @reactive.effect
    @reactive.event(input.button_check_format)
    def check_transcript_format():
        file = input.transcript()
        if not file:
            ui.modal_show(ui.modal(
                t("analysis", "modal_upload_transcript_first"),
                title=t("analysis", "modal_title_error"),
                easy_close=True,
                footer=ui.modal_button("OK", class_="btn-success"),
            ))
            return

        f = file[0]
        name = f.get("name", "transcript")
        ext = os.path.splitext(name)[1].lower()
        datapath = f["datapath"]

        if ext == ".pdf":
            ui.modal_show(ui.modal(
                t("analysis", "modal_format_pdf_unsupported"),
                title=t("analysis", "modal_title_format_check"),
                easy_close=True,
                footer=ui.modal_button(t("analysis", "modal_button_close"), class_="btn-success"),
            ))
            return

        if ext == ".docx":
            content = docx_to_json(datapath)
            if not isinstance(content, str):
                ui.modal_show(ui.modal(
                    t("analysis", "modal_format_docx_table"),
                    title=t("analysis", "modal_title_format_check"),
                    easy_close=True,
                    footer=ui.modal_button(t("analysis", "modal_button_close"), class_="btn-success"),
                ))
                return
            text = content
        else:
            text = read_txt(datapath)

        try:
            teacher = input.name_teacher()
        except Exception:
            teacher = None
        if is_valid_transcript_format(text, teacher):
            ui.modal_show(ui.modal(
                t("analysis", "modal_format_already_valid"),
                title=t("analysis", "modal_title_format_check"),
                easy_close=True,
                footer=ui.modal_button("OK", class_="btn-success"),
            ))
            return

        analysis = analyze_transcript(text, teacher)
        defaults = suggest_default_options(analysis, teacher)
        fmt_text.set(text)
        fmt_analysis.set(analysis)
        fmt_options.set(defaults)
        fmt_meta.set({"name": name, "ext": ext})
        _show_stage_speakers()

    def _read_speaker_mapping_from_inputs() -> ConversionOptions | None:
        analysis = fmt_analysis.get()
        options = fmt_options.get()
        if analysis is None or options is None:
            return None
        new_map: dict[str, str | None] = {}
        for i, raw in enumerate(analysis.speakers):
            try:
                val = input[f"fmt_spk_{i}"]()
            except Exception:
                val = None
            if val == "__ignore__" or not val:
                new_map[raw] = None
            else:
                new_map[raw] = val
        return ConversionOptions(
            speaker_map=new_map,
            strip_brackets=dict(options.strip_brackets),
            strip_tokens=dict(options.strip_tokens),
            teacher_label=options.teacher_label,
        )

    def _read_bracket_choices_into(options: ConversionOptions) -> ConversionOptions:
        analysis = fmt_analysis.get()
        if analysis is None:
            return options
        new_brackets = dict(options.strip_brackets)
        for grp in analysis.bracket_patterns:
            bid = _bracket_id(grp.delimiter)
            try:
                val = input[f"fmt_br_{bid}"]()
            except Exception:
                val = None
            if val is not None:
                new_brackets[grp.delimiter] = (val == "strip")
        new_tokens = dict(options.strip_tokens)
        for i, grp in enumerate(analysis.other_tokens):
            try:
                val = input[f"fmt_tok_{i}"]()
            except Exception:
                val = None
            if val is not None:
                new_tokens[grp.token] = (val == "strip")
        return ConversionOptions(
            speaker_map=options.speaker_map,
            strip_brackets=new_brackets,
            strip_tokens=new_tokens,
            teacher_label=options.teacher_label,
        )

    @reactive.effect
    @reactive.event(input.button_fmt_to_brackets)
    def _fmt_to_brackets():
        new_opts = _read_speaker_mapping_from_inputs()
        if new_opts is None:
            return
        fmt_options.set(new_opts)
        ui.modal_remove()
        _show_stage_brackets()

    @reactive.effect
    @reactive.event(input.button_fmt_back_speakers)
    def _fmt_back_speakers():
        options = fmt_options.get()
        if options is not None:
            fmt_options.set(_read_bracket_choices_into(options))
        ui.modal_remove()
        _show_stage_speakers()

    @reactive.effect
    @reactive.event(input.button_fmt_to_preview)
    def _fmt_to_preview():
        options = fmt_options.get()
        if options is None:
            return
        fmt_options.set(_read_bracket_choices_into(options))
        ui.modal_remove()
        _show_stage_preview()

    @reactive.effect
    @reactive.event(input.button_fmt_back_brackets)
    def _fmt_back_brackets():
        ui.modal_remove()
        analysis = fmt_analysis.get()
        if analysis is not None and (analysis.bracket_patterns or analysis.other_tokens):
            _show_stage_brackets()
        else:
            _show_stage_speakers()

    @render.download(filename=lambda: (converted_transcript.get() or {}).get("filename", "converted.txt"))
    def download_converted_transcript():
        data = converted_transcript.get()
        if data is None:
            return
        if data["ext"] == ".docx":
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp_path = tmp.name
            write_docx_from_text(tmp_path, data["text"])
            with open(tmp_path, "rb") as fh:
                yield fh.read()
        else:
            yield data["text"].encode("utf-8")
    

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
        return ui.div(
            ui.input_file(
                "codebook",
                t("analysis", "upload_codebook"),
                multiple=False,
                accept=[".txt", ".docx", ".pdf"],
                button_label=t("analysis", "browse"),
                placeholder=t("analysis", "placeholder"),
            ),
            **{"data-tt-help": t("onboarding", "tooltip_upload_codebook")},
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
        return ui.p(t("analysis", "preview_codebook"), class_="m-0 text-center")


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
        return ui.p(t("analysis", "preview_transcript"), class_="m-0 text-center")


    @render.ui
    def show_transcript_preview():
        if transcript_data.get() == None:
            return t("analysis", "placeholder_transcript")
        else:
            return transcript_data.get()

    # === section: testing ===

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
            accept=[".docx", ".xlsx", ".html", ".htm"],
            button_label=t("analysis", "browse"),
            placeholder=t("testing", "placeholder_report"),
        )

    @render.ui
    def loc_upload_report_b():
        return ui.input_file(
            "report_b",
            t("testing", "upload_report_b"),
            multiple=False,
            accept=[".docx", ".xlsx", ".html", ".htm"],
            button_label=t("analysis", "browse"),
            placeholder=t("testing", "placeholder_report"),
        )

    def _parse_uploaded_report(file_meta):
        try:
            return parse_report_impulses(file_meta['datapath']), None
        except ValueError as e:
            key = str(e)
            if key == "unsupported_format":
                return None, t("testing", "parse_error_unsupported_format")
            return None, t("testing", "parse_error_no_table")
        except Exception:
            return None, t("testing", "parse_error_no_table")

    @reactive.effect
    @reactive.event(input.report_a)
    def _process_report_a():
        f = input.report_a()
        if not f:
            return
        df, err = _parse_uploaded_report(f[0])
        report_a_df.set(df)
        report_a_error.set(err)

    @reactive.effect
    @reactive.event(input.report_b)
    def _process_report_b():
        f = input.report_b()
        if not f:
            return
        df, err = _parse_uploaded_report(f[0])
        report_b_df.set(df)
        report_b_error.set(err)

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

        pa = res.get("percent_agreement", float("nan"))
        alpha = res.get("krippendorff_alpha", float("nan"))
        pa_str = f"{pa * 100:.1f} %" if pa == pa else "n/a"
        alpha_str = f"{alpha:.3f}" if alpha == alpha else "n/a"
        items.append(
            ui.layout_columns(
                ui.value_box(t("testing", "summary_percent_agreement"),
                             pa_str, theme="info"),
                ui.value_box(t("testing", "summary_krippendorff"),
                             alpha_str, theme="info"),
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
        ci_low = res.get("kappa_ci_low", float("nan"))
        ci_high = res.get("kappa_ci_high", float("nan"))
        if ci_low == ci_low and ci_high == ci_high:
            ci_text = f" [{ci_low:.3f}, {ci_high:.3f}]"
            ci_caption = t("testing", "kappa_ci_label")
            value_html = ui.tags.div(
                f"κ = {k:.3f}",
                ui.tags.span(ci_text, style="font-size: 1.4rem; font-weight: 400; color: var(--bs-secondary-color); margin-left: 0.5rem;"),
                ui.tags.span(f" ({ci_caption})", style="font-size: 0.9rem; color: var(--bs-secondary-color);"),
                style="font-size: 2.4rem; font-weight: 600;",
            )
        else:
            value_html = ui.tags.div(
                f"κ = {k:.3f}",
                style="font-size: 2.4rem; font-weight: 600;",
            )
        return ui.TagList(
            value_html,
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

    @render.ui
    def testing_per_code_header():
        return t("testing", "per_code_header")

    @render.ui
    def testing_per_code_table():
        res = _agreement()
        if res is None:
            return ui.p(t("testing", "kappa_not_ready"))
        per_code = res.get("per_code")
        if per_code is None or per_code.empty:
            return ui.p("—")
        header = ui.tags.thead(ui.tags.tr(
            ui.tags.th(t("testing", "per_code_col_code")),
            ui.tags.th(t("testing", "per_code_col_n_a")),
            ui.tags.th(t("testing", "per_code_col_n_b")),
            ui.tags.th(t("testing", "per_code_col_f1")),
            ui.tags.th(t("testing", "per_code_col_precision")),
            ui.tags.th(t("testing", "per_code_col_recall")),
        ))
        body_rows = []
        for _, row in per_code.iterrows():
            body_rows.append(ui.tags.tr(
                ui.tags.th(str(row["Code"])),
                ui.tags.td(str(int(row["n(A)"]))),
                ui.tags.td(str(int(row["n(B)"]))),
                ui.tags.td(f"{row['F1']:.3f}"),
                ui.tags.td(f"{row['Precision']:.3f}"),
                ui.tags.td(f"{row['Recall']:.3f}")))
        body = ui.tags.tbody(*body_rows)
        return ui.tags.table(header, body,
                             class_="table table-sm table-bordered table-striped")

    @render.ui
    def testing_export_button():
        if _agreement() is None:
            return None
        return ui.download_button(
            "download_testing_report",
            t("testing", "export_report"),
            icon=icon_svg("download"),
            class_="btn-sm",
        )

    @render.download(filename=lambda: f"{date.today().isoformat()} - Intercoder Agreement.xlsx")
    def download_testing_report():
        res = _agreement()
        if res is None:
            ui.notification_show(t("testing", "no_data"), type="warning", duration=4)
            return None
        suffix = ".xlsx"
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.close()
        try:
            export_testing_agreement(
                tmp_file.name, res,
                sheet_overview=t("report_options", "sheet_overview"),
                sheet_confusion=t("testing", "confusion_header"),
                sheet_per_code=t("testing", "per_code_header"),
                sheet_pairs="Pairs",
            )
        except RuntimeError as e:
            key = str(e)
            msg = t("report_options", key) if key == "xlsx_unavailable" else str(e)
            ui.notification_show(msg, type="error", duration=6)
            return None
        return tmp_file.name



    # === section: results ===

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
                ui.tags.hr(),
                ui.p(t("onboarding", "empty_results_message")),
                ui.input_action_button(
                    "tt_demo_open_from_modal",
                    t("onboarding", "demo_button"),
                    icon=icon_svg("vial"),
                    class_="btn-primary btn-sm",
                ),
                title=t("results", "no_results_title"),
                easy_close=True,
                footer=ui.modal_button("OK", class_="btn-success"),
                size="m",
            )
            ui.modal_show(m)
            ui.update_navs("main_tabs", selected='<div id="loc_title_analysis" class="shiny-text-output"></div>')

    # Anzeige der allgemeinen Informationen
    @render.ui
    def loc_quantitative_analysis():
        return ui.span(t("results", "section_quantitative_analysis"))


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
        # Map each speaker row to teacher or students using the user-provided name.
        teacher_label = t("stats", "teacher")
        students_label = t("stats", "students")
        teacher_name = input.name_teacher() or t("analysis", "name_teacher_var")
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
    @render.plot(alt="placeholder", height=260)
    def sim_stats_plot():
        if analysis_state.get() == False:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        else:
            return make_sim_stats_plot()


    @render.ui
    def loc_over_time_quant_title():
        return ui.span(t("results", "over_time_quant_title"))


    def _segment_labels_for(n_segments):
        if n_segments == 3:
            return [t("results", "section_first"),
                    t("results", "section_middle"),
                    t("results", "section_last")]
        return [f"{t('results', 'section')} {i + 1}" for i in range(n_segments)]


    @reactive.calc
    def make_sim_stats_over_time_plot():
        req(transcript_data.get() is not None)
        transcript = transcript_data.get()
        teacher = input.name_teacher() or t("analysis", "name_teacher_var")
        n_segments = 3
        df = dialog_stats_over_time(
            transcript, teacher,
            n_segments=n_segments,
            segment_labels=_segment_labels_for(n_segments),
        )
        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        pivot = df.pivot(index="Abschnitt", columns="Sprecher_Gruppe", values="Wörter") \
                  .reindex(_segment_labels_for(n_segments))
        ax = pivot.plot(kind='bar', rot=0, alpha=1)
        ax.set_xlabel(t("results", "section"))
        ax.set_ylabel(t("results", "words_total"))
        ax.set_axisbelow(True)
        ax.grid(color='gray', axis='y')
        ax.legend(loc="upper right", fontsize=8, title=None)
        for container in ax.containers:
            ax.bar_label(container, label_type='edge', fontsize=8)
        return ax.get_figure()


    @render.plot(alt="placeholder", height=240)
    def sim_stats_over_time_plot():
        if not analysis_state.get():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        return make_sim_stats_over_time_plot()

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
        return ui.span(t("results", "section_qualitative_analysis"))


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
            # Exclude uncoded turns (empty Shortcode from the LEFT JOIN in
            # make_qualitative_stats_df) — otherwise "" usually wins the mode().
            codes = df[t("report", "shortcode")].astype(str).str.strip()
            codes = codes[codes != ""]
            if codes.empty:
                return t("system_prompts", "no_code")
            most_used_codes = codes.mode().to_list()
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
    @render.plot(alt="Noch keine Daten", height=260)
    def qualitative_stats_plot():
        if not llm_analysis_data.get():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        else:
            return make_qualitative_stats_plot()


    @render.ui
    def loc_over_time_quali_title():
        return ui.span(t("results", "over_time_quali_title"))


    @reactive.calc
    def make_qualitative_stats_over_time_plot():
        req(llm_analysis_data.get())
        req(transcript_data.get() is not None)
        latest_df = llm_analysis_data.get()[-1]
        if latest_df is None or latest_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        transcript = transcript_data.get()
        teacher = input.name_teacher() or t("analysis", "name_teacher_var")
        n_segments = 3
        labels = _segment_labels_for(n_segments)
        mapped = map_impulses_to_turn_index(latest_df, transcript, teacher)
        total_turns = count_transcript_turns(transcript, teacher)
        dist = code_distribution_over_time(
            mapped, total_turns,
            n_segments=n_segments,
            segment_labels=labels,
        )
        if dist.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        pivot = (dist.pivot(index="Abschnitt", columns="Shortcode", values="Anteil")
                     .fillna(0)
                     .reindex(labels))
        ax = pivot.plot(kind='bar', stacked=True, rot=0, alpha=1)
        ax.set_xlabel(t("results", "section"))
        ax.set_ylabel(t("results", "share"))
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True)
        ax.grid(color='gray', axis='y')
        ax.legend(loc="upper right", fontsize=8, title=t("report", "shortcode"),
                  bbox_to_anchor=(1.0, 1.0))
        return ax.get_figure()


    @render.plot(alt="placeholder", height=240)
    def qualitative_stats_over_time_plot():
        if not llm_analysis_data.get():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        return make_qualitative_stats_over_time_plot()


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

        transcript_text = transcript_data.get()
        if not transcript_text:
            # Fallback: converted transcript (format wizard result)
            conv = converted_transcript.get()
            if conv and conv.get("text"):
                transcript_text = conv["text"]
        if transcript_text:
            teacher_name = input.name_teacher() or t("analysis", "name_teacher_var")
            turns = _parse_turns(transcript_text, teacher_name)
            all_turns_df = pd.DataFrame(turns, columns=["Sprecher", "Impuls"])
            # Normalize parsed speaker to canonical teacher_name (case-insensitive
            # regex may produce the verbatim transcript casing, e.g. "Lehrer" vs "LEHRER").
            all_turns_df["Sprecher"] = all_turns_df["Sprecher"].apply(
                lambda s: teacher_name if s.lower() == teacher_name.lower() else s
            )
            all_turns_df['#'] = range(1, len(all_turns_df) + 1)
            # merge key to avoid ambiguous matches on duplicate utterance texts
            all_turns_df["__key__"] = all_turns_df["Sprecher"] + " :: " + all_turns_df["Impuls"]
            coded = analysis_df[["Sprecher", "Impuls", "Shortcode"]].copy()
            # Normalize teacher speaker name: LLMs sometimes return "Lehrperson" or
            # "Lehrer" even when the transcript uses the configured teacher_name (e.g.
            # "LEHRER"). Map any case-insensitive match to the canonical name so the
            # join key aligns with all_turns_df.
            _teacher_aliases = {"lehrperson", "lehrer", "lehrkraft", teacher_name.lower()}
            coded["Sprecher"] = coded["Sprecher"].apply(
                lambda s: teacher_name if str(s).lower() in _teacher_aliases else s
            )
            coded["__key__"] = coded["Sprecher"] + " :: " + coded["Impuls"]
            coded = coded.drop_duplicates(subset=["__key__"], keep="first")
            merged = pd.merge(
                all_turns_df,
                coded[["__key__", "Shortcode"]],
                on="__key__",
                how="left",
            )
            merged = merged.drop(columns=["__key__"])
            merged = merged[['#', 'Sprecher', 'Impuls', 'Shortcode']].copy()
            merged["Shortcode"] = merged["Shortcode"].fillna("").astype(str)
            merged.columns = cols
            qual_stats_df.set(merged)
            return merged
        else:
            # Fallback: just coded impulses (no transcript available)
            analysis_df['#'] = analysis_df.reset_index().index + 1
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



    # === section: options ===

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
        key_for = {
            "openai": "api_key_openai",
            "groq": "api_key_groq",
            "anthropic": "api_key_anthropic",
            "ollama": "api_key_ollama",
        }
        target_for = {
            "openai": api_key_openai,
            "groq": api_key_groq,
            "anthropic": api_key_anthropic,
            "ollama": api_key_ollama,
        }
        if selected in key_for:
            persisted = safe_set_password("talktrace", key_for[selected], input.api_key())
            target_for[selected].set(input.api_key())
            if not persisted:
                ui.notification_show(
                    t("options", "keyring_unavailable"),
                    type="warning",
                    duration=8,
                )
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
        key_for = {
            "openai": "api_key_openai",
            "groq": "api_key_groq",
            "anthropic": "api_key_anthropic",
            "ollama": "api_key_ollama",
        }
        target_for = {
            "openai": api_key_openai,
            "groq": api_key_groq,
            "anthropic": api_key_anthropic,
            "ollama": api_key_ollama,
        }
        if selected in key_for:
            safe_delete_password("talktrace", key_for[selected])
            target_for[selected].set(None)
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

        def _parse_cost(raw: str) -> float:
            try:
                return float(raw.strip().replace(",", "."))
            except (ValueError, AttributeError):
                return 0.0

        input_cost = _parse_cost(input.intput_cost())
        output_cost = _parse_cost(input.output_cost())

        config.add_model(input.model_provider(), input.model_id(), input_cost, output_cost)
        available_models = config.get_models()
        model_deleted.set(model_deleted.get() + 1)
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

