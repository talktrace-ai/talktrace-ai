"""Analysis pipeline: button, run_analysis coroutine, async kickoff, status text.

Sets ``state.run_analysis`` so other handlers (e.g. demo loaders) can trigger
the same coroutine. Reads ``state.effective_*_prompt``, ``state._speaker_flags``
(from _prompts) and bumps ``state.history_version`` (from _session) on
successful LLM auto-save.
"""
import time

from .._common import *


def register(state):
    input = state.input
    t = state.t
    config = state.config
    transcript_data = state.transcript_data
    codebook_data = state.codebook_data
    api_key_groq = state.api_key_groq
    api_key_openai = state.api_key_openai
    api_key_anthropic = state.api_key_anthropic
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
    code_legend_storage = state.code_legend_storage

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
    async def run_analysis():
        req(transcript_data.get() != None)
        # If teacher analysis is desired, verify the name exists in the transcript.
        teacher_name = input.name_teacher()
        transcript = transcript_data.get()
        teacher_on, students_on = state._speaker_flags()
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
        if input.llm_switch():
            # Codebuch über den geteilten State prüfen, nicht über den Datei-
            # Widget-Wert. Bei Demo-Daten oder Session-Restore ist das Widget
            # leer, der State aber gefüllt — req(input.codebook()) würde dort
            # mit einer leeren SilentException abbrechen.
            req(codebook_data.get() is not None)
            teacher_on, students_on = state._speaker_flags()
            req(teacher_on or students_on)
            sys_p = state.effective_system_prompt()
            usr_p = state.effective_user_prompt()
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

            await reactive.flush()

        # Total für den Progress-Indikator: Summe aller codierbaren Beiträge
        # (Lehrperson + Schüler:innen) aus dem soeben berechneten stats-DF.
        try:
            total_impulses = int(df_stats['Anzahl_Beitraege'].sum()) if df_stats is not None else 0
        except Exception:
            total_impulses = 0

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
                # 10-Punkt-Stepper unter dem Analyze-Button initialisieren.
                analysis_progress.set((0, total_impulses))
                # Switch zum Results-Tab schon jetzt, damit der User die
                # ankommenden Items sieht.
                ui.update_navset("main_tabs", selected='<span class="shiny-html-output" id="loc_title_results"></span>')
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
                            analysis_progress.set((len(working_items), total_impulses))
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
                    analysis_progress.set(None)
                    await reactive.flush()
                return f"{t('system_prompts', 'error')}: {error_msg}. {t('system_prompts', 'try_again')}"

            if not working_items:
                async with reactive.lock():
                    existing_data.pop()
                    llm_analysis_data.set(list(existing_data))
                    analysis_progress.set(None)
                    await reactive.flush()
                return f"{t('system_prompts', 'error')}: LLM returned 0 coded items. {t('system_prompts', 'try_again')}"

            # Erfolg: Bar auf 10/10 stehen lassen, bis nächster Run / Reset.
            async with reactive.lock():
                final_total = total_impulses if total_impulses > 0 else len(working_items)
                analysis_progress.set((final_total, final_total))
                await reactive.flush()

            print(f"[LLM ANALYSIS streaming] provider={config.get_current_api()} model={model.get()} returned {len(working_items)} coded items")
            did_llm_analysis = True
        # Mark Analysis as Completed
        async with reactive.lock():
            analysis_state.set(True)
            try:
                current_tab = input.main_tabs()
            except Exception:
                current_tab = None
            mark_tab_unread(state.tab_badge_results, current_tab, "loc_title_results")
            await reactive.flush()

        # Auto-save to history after a successful LLM analysis. We only persist
        # when the LLM actually ran (not LLM-off quick stats), since those are
        # not the kind of result the user wants to revisit.
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
                    state.history_version.set(state.history_version.get() + 1)
                    await reactive.flush()
            except Exception as exc:
                print(f"[history] auto-save after LLM analysis failed: {exc}")

        # Automatically Switch to Results Tab
        async with reactive.lock():
            ui.update_navset("main_tabs", selected='<div id="loc_title_results" class="shiny-text-output"></div>')
            await reactive.flush()
        # Erfolgsfall: kein Status-Text mehr — der Bar (im Streaming-Pfad) bzw.
        # der Tab-Wechsel (Nicht-Streaming) ist das Feedback. Status-Text ist
        # ab jetzt ausschließlich Fehlermeldungen vorbehalten.
        return ""

    state.run_analysis = run_analysis

    # Status-Text + Trigger entkoppelt vom Output-Renderer. Die Analyse
    # läuft in einer eigenen asyncio-Task: nur so wird der Reactive-Lock
    # zwischen Batches freigegeben, sodass abhängige Outputs (Tabelle,
    # Plots, Header auf dem Results-Tab) progressiv neu rendern können.
    # Liefe run_analysis direkt im Effect oder im Output, hielte Shiny den
    # Lock für die gesamte Coroutine — die UI bliebe bis zum Schluss auf
    # "Ladesymbol", egal wie oft wir intern .set()/flush() aufrufen.
    analysis_status_msg = reactive.value("")
    # 10-Punkt-Stepper-Zustand: None = nichts anzeigen, sonst (current, total).
    # Wird aus run_analysis() gesetzt; vom Reset-/Restore-Handler auf None
    # zurückgesetzt (siehe _session.py).
    analysis_progress = reactive.value(None)
    state.analysis_progress = analysis_progress

    async def _run_analysis_async():
        msg = ""
        try:
            msg = await run_analysis()
        except Exception as e:
            # Shiny's req() raises a SilentException with no message when an
            # input is missing — that's a control-flow signal, not a user-
            # facing error. Suppressing it avoids the empty "Error: " banner.
            err_text = str(e).strip()
            if not err_text or e.__class__.__name__ == "SilentException":
                msg = ""
            else:
                msg = f"Error: {err_text}"
            print(f"[analysis] task failed: {e!r}")
        async with reactive.lock():
            analysis_status_msg.set(msg or "")
            # Bei Fehler den Bar verstecken; bei Erfolg ist er bereits auf 10/10
            # gesetzt und soll stehenbleiben.
            if msg:
                analysis_progress.set(None)
            await reactive.flush()

    @reactive.effect
    @reactive.event(input.button_analysis)
    def _kick_off_analysis():
        # Beim Klick alten Fehlertext / Bar-Reststand verwerfen.
        analysis_status_msg.set("")
        analysis_progress.set(None)
        asyncio.create_task(_run_analysis_async())

    # Stepper-Granularität: jeder Punkt entspricht (100 / DOT_COUNT) %.
    DOT_COUNT = 20

    def _filled_dot_count(current: int, total: int) -> int:
        # Rundet zur nächsten Stufe, kappt aber bei (DOT_COUNT - 1) bis das
        # Streaming wirklich durch ist (current >= total) — so leuchtet der
        # letzte Punkt erst beim expliziten "fertig"-Set.
        if total <= 0:
            return 0
        if current >= total:
            return DOT_COUNT
        return min(DOT_COUNT - 1, round(current / total * DOT_COUNT))

    @render.ui
    def start_analysis():
        progress = analysis_progress.get()
        msg = analysis_status_msg.get()
        if progress is not None:
            current, total = progress
            filled = _filled_dot_count(current, total)
            completed = current >= total and total > 0
            dots = [
                ui.tags.span(class_=f"tt-dot{' filled' if i < filled else ''}")
                for i in range(DOT_COUNT)
            ]
            inner = [ui.div(*dots, class_="tt-progress-dots")]
            if completed:
                inner.append(ui.div(
                    "✓ ", t("sidebar", "analysis_completed"),
                    class_="tt-progress-label completed",
                ))
            return ui.div(*inner, class_="tt-progress-wrap")
        if msg:
            return ui.div(msg, class_="tt-analysis-error")
        return None
