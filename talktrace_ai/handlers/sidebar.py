"""Sidebar section: model/provider select, run_analysis, report download, history, reset."""
import time

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
    code_legend_storage = state.code_legend_storage
    estimated_cost = state.estimated_cost
    token_count = state.token_count
    system_prompt = state.system_prompt
    user_prompt = state.user_prompt

    ### Sidebar --------------------------------------------------------
    # Model Selection
    @render.ui
    def loc_dynamic_model_select():
        return ui.div(
            ui.input_select("provider_select", t("sidebar", "provider_select"), choices={"openai": "OpenAI", "groq": "Groq", "anthropic": "Anthropic", "ollama": "Ollama"}, selected=config.get_current_api()),
            ui.input_select("model_select", t("sidebar", "model_select"), choices=state.select_api_choices(), selected=config.get_current_model()),
            **{"data-tt-help": t("onboarding", "tooltip_model_select")},
        )

    # Hinweis nur bei aktivem Ollama-Provider — Ollama Cloud ist kostenlos und
    # gut zum Testen, aber Latenz schwankt stark; ein dezenter Hinweis spart
    # Frust ohne den User mit einem Modal zu nerven.
    @render.ui
    def loc_ollama_hint():
        try:
            if input.provider_select() != "ollama":
                return None
        except Exception:
            return None
        return ui.tags.p(t("sidebar", "ollama_cloud_hint"), class_="text-muted small")


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
        available_models = state.select_api_choices()
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
            ui.input_switch("multi_coding_switch", t("sidebar", "multi_coding_switch"), False),
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

    def _multi_coding_flag() -> bool:
        """Schalter-Wert defensiv lesen — der Switch wird nur gerendert,
        solange ``llm_switch`` aktiv ist. Default OFF.
        """
        try:
            return bool(input.multi_coding_switch())
        except Exception:
            return False

    def _multi_coding_suffix(kind: str = "system") -> str:
        """Liefert den Prompt-Zusatz, der dem LLM mitteilt, ob Mehrfach-
        Codierung erlaubt/erwünscht (ON) oder verboten (OFF) ist. Wird
        sowohl an System- als auch an User-Prompt angehängt, damit das
        Modell unmissverständlich weiß, was es tun soll. Post-Processing
        (Hierarchie + drop_duplicates / groupby) bleibt als Sicherheitsnetz."""
        prefix = "user_prompt_multi_coding" if kind == "user" else "prompt_multi_coding"
        key = f"{prefix}_{'on' if _multi_coding_flag() else 'off'}"
        return t("sidebar", key)

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
        return base + _speaker_filter_suffix("system") + _multi_coding_suffix("system")

    @reactive.calc
    def effective_user_prompt():
        teacher, students = _speaker_flags()
        raw = _sanitize_prompt_for_speakers(user_prompt.get(), teacher, students)
        # Beide Instruktions-Suffixe (Sprecher-Filter + Multi-Coding) werden
        # gemeinsam direkt nach dem {transcript}-Block platziert. Hintergrund:
        # LLMs leiden bei sehr langen Kontexten unter "lost in the middle" —
        # Anweisungen über Output-Format und Filter müssen nahe am Transkript
        # sitzen, nicht am Ende nach tausenden Token Codebook.
        combined = _speaker_filter_suffix("user") + _multi_coding_suffix("user")
        if not combined:
            return raw
        if "{transcript}" in raw:
            target = "{transcript}"
            idx = raw.index(target)
            insert_pos = idx + len(target)
            return raw[:insert_pos] + "\n\n" + combined + raw[insert_pos:]
        return raw + combined

    state.effective_system_prompt = effective_system_prompt
    state.effective_user_prompt = effective_user_prompt


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
        return ui.input_action_button(
            "button_analysis",
            t("sidebar", "button_analysis"),
            icon=icon_svg("magnifying-glass-chart"),
            class_="btn-success",
        )

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
            stream_gen_args = None  # populated in streaming mode (see below)
            streaming_enabled = config.get_advanced().get("streaming", False)
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

                if streaming_enabled:
                    # Streaming-Pfad: Generator-Args zwischenspeichern, Ausführung
                    # erfolgt sequentiell nach der Stats-Berechnung. Items werden
                    # progressiv in den DataFrame geschoben.
                    lang = config.get_localization().get("current_language", "de")
                    if current_api == "groq":
                        req(api_key_groq.get() != None)
                        client = get_groq_client(api_key_groq.get())
                        stream_gen_args = (
                            llm_analysis_groq_stream,
                            (sys_p, usr_p, mdl, transcript, cb, client),
                            {"language": lang},
                        )
                    elif current_api == "openai":
                        req(api_key_openai.get() != None)
                        client = get_openai_client(api_key_openai.get())
                        stream_gen_args = (
                            llm_analysis_openai_stream,
                            (sys_p, usr_p, mdl, transcript, cb, client),
                            {},
                        )
                    elif current_api == "anthropic":
                        req(api_key_anthropic.get() != None)
                        client = get_anthropic_client(api_key_anthropic.get())
                        stream_gen_args = (
                            llm_analysis_anthropic_stream,
                            (sys_p, usr_p, mdl, transcript, cb, client),
                            {},
                        )
                    elif current_api == "ollama":
                        stream_gen_args = (
                            llm_analysis_ollama_stream,
                            (sys_p, usr_p, mdl, transcript, cb),
                            {"language": lang},
                        )
                else:
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

            # Alle Reactive-Sets in einem lock+flush-Block, damit sie als
            # zusammenhängender Snapshot ans UI gehen — sonst sieht der User
            # die quantitativen Ergebnisse erst, wenn die ganze Analyse fertig
            # ist (Reactive-Updates aus einer Task werden ohne explizites
            # Flushen nicht weitergereicht).
            async with reactive.lock():
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
                await reactive.flush()

            did_llm_analysis = False
            # Auf LLM-Resultat warten, falls aktiviert.
            if llm_task is not None:
                llm_response = await llm_task

                if llm_response is None:
                    llm_response = json.dumps({"error": "No API provider matched or no response received."})

                if '"error":' in llm_response:
                    return f"{t("system_prompts", "error")}: {json.loads(llm_response)['error']}. {t("system_prompts", "try_again")}"

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

                async with reactive.lock():
                    existing_data = llm_analysis_data.get()
                    existing_data.append(new_data_df)
                    llm_analysis_data.set(list(existing_data)) # Important to Set as a List to Avoid Reactivity Issues, Due to Immutability Logic of Python!!!
                    analysis_llm_state.set(True)
                    await reactive.flush()
                did_llm_analysis = True
            elif stream_gen_args is not None:
                # Streaming-Pfad: progressive UI-Updates via async-Generator.
                # Throttling vermeidet Reactivity-Thrash bei vielen Items.
                # Jeder reactive-set-Block läuft in einem eigenen
                # `async with reactive.lock(): ... ; await reactive.flush()`,
                # damit der Lock zwischen Batches freigegeben wird und andere
                # Outputs (Tabelle, Plots, Header) progressiv rendern können.
                fn, args, kwargs = stream_gen_args

                async with reactive.lock():
                    existing_data = llm_analysis_data.get()
                    empty_df = pd.DataFrame(columns=['#', "Sprecher", "Shortcode", "Impuls"])
                    existing_data.append(empty_df)
                    llm_analysis_data.set(list(existing_data))
                    analysis_llm_state.set(True)
                    # analysis_state schon jetzt setzen, damit die Results-Renderer
                    # nicht weiter auf "Ladesymbol" stehen bleiben — sie sind alle
                    # mit req(analysis_state.get()) gegated. Im Streaming-Modus
                    # bedeutet das Flag "Daten kommen rein", nicht "fertig".
                    analysis_state.set(True)
                    # Switch zum Results-Tab schon jetzt, damit der User die
                    # ankommenden Items sieht.
                    ui.update_navs("main_tabs", selected='<div id="loc_title_results" class="shiny-text-output"></div>')
                    await reactive.flush()

                working_items = []
                last_update = time.monotonic()
                error_msg = None
                THROTTLE_S = 0.2
                BATCH = 3
                pending = 0
                items_since_flush = 0

                async for event in async_stream(fn, *args, **kwargs):
                    etype = event.get("type")
                    if etype == "item":
                        working_items.append(event["data"])
                        pending += 1
                        items_since_flush += 1
                        now = time.monotonic()
                        if pending >= BATCH or (now - last_update) >= THROTTLE_S:
                            async with reactive.lock():
                                df = pd.DataFrame(working_items, columns=['#', "Sprecher", "Shortcode", "Impuls"])
                                existing_data[-1] = df
                                llm_analysis_data.set(list(existing_data))
                                await reactive.flush()
                            pending = 0
                            last_update = now
                    elif etype == "done":
                        # raw_json is already cached inside the provider on
                        # success. Nothing to do here besides flushing.
                        pass
                    elif etype == "error":
                        error_msg = event.get("message", "Unknown streaming error")
                        break

                # Final flush of any remaining items.
                async with reactive.lock():
                    df = pd.DataFrame(working_items, columns=['#', "Sprecher", "Shortcode", "Impuls"])
                    existing_data[-1] = df
                    llm_analysis_data.set(list(existing_data))
                    await reactive.flush()

                if error_msg and not working_items:
                    async with reactive.lock():
                        existing_data.pop()
                        llm_analysis_data.set(list(existing_data))
                        await reactive.flush()
                    return f"{t('system_prompts', 'error')}: {error_msg}. {t('system_prompts', 'try_again')}"

                if not working_items:
                    async with reactive.lock():
                        existing_data.pop()
                        llm_analysis_data.set(list(existing_data))
                        await reactive.flush()
                    return f"{t('system_prompts', 'error')}: LLM returned 0 coded items. {t('system_prompts', 'try_again')}"

                print(f"[LLM ANALYSIS streaming] provider={config.get_current_api()} model={model.get()} returned {len(working_items)} coded items")
                did_llm_analysis = True
            p.set(4, message=t("sidebar", "analysis_completed"))
            # Mark Analysis as Completed
            async with reactive.lock():
                analysis_state.set(True)
                await reactive.flush()

        # Auto-save to history after a successful LLM analysis. We only persist
        # when the LLM actually ran (not for force_no_llm demo loads or LLM-off
        # quick stats), since those are not the kind of result the user wants
        # to revisit.
        if did_llm_analysis and stats.get() is not None:
            try:
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
                    n_turns = int(stats.get()['Anzahl_Beitraege'].sum())
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
                async with reactive.lock():
                    history_version.set(history_version.get() + 1)
                    await reactive.flush()
            except Exception as exc:
                print(f"[history] auto-save after LLM analysis failed: {exc}")

        # Automatically Switch to Results Tab
        async with reactive.lock():
            ui.update_navs("main_tabs", selected='<div id="loc_title_results" class="shiny-text-output"></div>')
            await reactive.flush()
        return t("sidebar", "analysis_completed")

    state.run_analysis = run_analysis

    # Status-Text + Trigger entkoppelt vom Output-Renderer. Die Analyse
    # läuft in einer eigenen asyncio-Task: nur so wird der Reactive-Lock
    # zwischen Batches freigegeben, sodass abhängige Outputs (Tabelle,
    # Plots, Header auf dem Results-Tab) progressiv neu rendern können.
    # Liefe run_analysis direkt im Effect oder im Output, hielte Shiny den
    # Lock für die gesamte Coroutine — die UI bliebe bis zum Schluss auf
    # "Ladesymbol", egal wie oft wir intern .set()/flush() aufrufen.
    analysis_status_msg = reactive.value("")

    async def _run_analysis_async():
        msg = ""
        try:
            msg = await run_analysis()
        except Exception as e:
            msg = f"Error: {e}"
            print(f"[analysis] task failed: {e}")
        async with reactive.lock():
            analysis_status_msg.set(msg or "")
            await reactive.flush()

    @reactive.effect
    @reactive.event(input.button_analysis)
    def _kick_off_analysis():
        asyncio.create_task(_run_analysis_async())

    @render.text
    def start_analysis():
        return analysis_status_msg.get()

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
                plot_ot_quant = state.make_sim_stats_over_time_plot()
                teacher_name = input.name_teacher() or t("analysis", "name_teacher_var")
                df_ot_quant = dialog_stats_over_time(
                    transcript_data.get(), teacher_name,
                    n_segments=3, segment_labels=state.segment_labels_for(3),
                )
            except Exception as e:
                print(f"[REPORT] over-time quant plot failed: {e}")
        if sections.get("over_time_quali") and has_llm and transcript_data.get() is not None:
            try:
                plot_ot_quali = state.make_qualitative_stats_over_time_plot()
                latest_df = llm_analysis_data.get()[-1] if llm_analysis_data.get() else None
                if latest_df is not None:
                    teacher_name = input.name_teacher() or t("analysis", "name_teacher_var")
                    mapped = map_impulses_to_turn_index(latest_df, transcript_data.get(), teacher_name)
                    total_turns = count_transcript_turns(transcript_data.get(), teacher_name)
                    df_ot_quali = code_distribution_over_time(
                        mapped, total_turns, n_segments=3, segment_labels=state.segment_labels_for(3),
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
