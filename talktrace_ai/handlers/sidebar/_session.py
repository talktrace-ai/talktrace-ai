"""Session import/export, history (save/load/delete), reset.

Owns ``state.history_version`` which ``_analysis`` bumps after auto-save.
"""
from .._common import *


def register(state):
    input = state.input
    t = state.t
    config = state.config
    transcript_data = state.transcript_data
    codebook_data = state.codebook_data
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
    teacher_impulses_count = state.teacher_impulses_count
    analysis_state = state.analysis_state
    analysis_llm_state = state.analysis_llm_state
    sim_plot = state.sim_plot
    qual_plot = state.qual_plot
    qual_stats_df = state.qual_stats_df
    placeholder_plot = state.placeholder_plot
    code_legend_storage = state.code_legend_storage

    # Eine bereits gespeicherte Sitzung wiederherstellen, ohne run_analysis
    # erneut aufzurufen — sonst würde im schlimmsten Fall ein neuer LLM-Call
    # ausgelöst (Geld!) und in jedem Fall die Statistik unnötig neu berechnet,
    # obwohl wir sie bereits aus dem Pickle haben.
    def _restore_session_state(session_data):
        teacher_name = input.name_teacher() or t("analysis", "name_teacher_var")

        transcript_data.set(session_data.get("transcript_data"))
        num_participants.set(session_data.get("num_participants"))
        participation_rate.set(session_data.get("participation_rate"))
        stats.set(session_data.get("stats"))
        llm_analysis_data.set(session_data.get("llm_analysis_data") or [])
        analysis_llm_state.set(bool(session_data.get("analysis_llm_state")))
        code_legend_storage.set(session_data.get("code_legend_storage") or "")
        if "placeholder_plot" in session_data:
            placeholder_plot.set(session_data.get("placeholder_plot"))

        # Abgeleitete Werte aus dem stats-DataFrame wieder ableiten — diese
        # sind nicht im Pickle (Backward-Compat mit älteren Exports), aber
        # alle Information dafür steckt in stats + transcript.
        df_stats = session_data.get("stats")
        if df_stats is not None and not df_stats.empty:
            def _safe(speaker, col, default=0):
                m = df_stats.loc[df_stats['Sprecher'] == speaker, col]
                return m.values[0] if not m.empty else default

            t_turns.set(_safe(teacher_name, 'Anzahl_Beitraege'))
            t_turns_length.set(round(_safe(teacher_name, 'Durchschnitt_Woerter'), 1))
            t_turns_length_mean_sd.set(round(_safe(teacher_name, 'Median_Woerter'), 1))
            p_turns.set(_safe("Schüler:innen", 'Anzahl_Beitraege'))
            p_turns_length.set(round(_safe("Schüler:innen", 'Durchschnitt_Woerter'), 1))
            p_turns_length_mean_sd.set(_safe("Schüler:innen", 'Median_Woerter'))
            teacher_impulses_count.set(count_teacher_impulses(df_stats, teacher_name))

        transcript = session_data.get("transcript_data")
        if transcript:
            try:
                stats_per_speaker.set(dialog_stats_per_speaker(transcript, teacher_name))
            except Exception as exc:
                print(f"[restore] dialog_stats_per_speaker failed: {exc}")

        analysis_state.set(True)
        # Restore-Pfade switchen direkt zum Results-Tab — Badge gleich auf
        # "read" setzen, damit kein roter Punkt aufblitzt.
        state.tab_badge_results.set("read")
        # Geladene Sessions sind kein laufender Streaming-Run — den 10-Punkt-Bar
        # nicht stehenlassen (semantisch heißt er "dieser Run ist gerade durchgelaufen").
        if hasattr(state, "analysis_progress"):
            state.analysis_progress.set(None)
        # Restored sessions are by definition completed runs — clear any
        # leftover cancellation banner from the export.
        if hasattr(state, "analysis_cancelled"):
            state.analysis_cancelled.set(False)
        ui.update_switch("llm_switch", value=False)

    # Import Session
    @render.ui
    def loc_button_import_session():
        return ui.input_file("button_import_session", t("sidebar", "import_session"), accept=[".pkl"], multiple=False, placeholder=t("analysis", "placeholder"), button_label=t("analysis", "browse")),


    @reactive.effect
  #  @reactive.event(input.button_import_session)
    def button_import_session():

        file = input.button_import_session()

        if not file:
            return

        try:
            with open(file[0]["datapath"], "rb") as f:
                session_data = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as exc:
            print(f"[import] failed to read .pkl: {exc}")
            return

        with reactive.isolate():
            _restore_session_state(session_data)

        ui.update_navset("main_tabs", selected='<span class="shiny-html-output" id="loc_title_results"></span>')

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
    state.history_version = history_version


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
    def load_history_selected():
        fname = input.history_select()
        if not fname:
            return
        try:
            session_data = load_history_entry(fname)
        except (OSError, pickle.UnpicklingError) as exc:
            print(f"[history] load failed: {exc}")
            return
        with reactive.isolate():
            _restore_session_state(session_data)
        ui.modal_remove()
        ui.update_navset("main_tabs", selected='<span class="shiny-html-output" id="loc_title_results"></span>')


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
        # Indikatoren zurücksetzen, sonst zeigen sie veraltete Infos der
        # gerade gelöschten Session: Format-Status (Wand-Button) und die
        # Tab-Badges an Results/Testing.
        state.transcript_format_status.set(None)
        state.tab_badge_results.set(None)
        state.tab_badge_testing.set(None)
        # 10-Punkt-Bar löschen — neue Session, neuer Run.
        if hasattr(state, "analysis_progress"):
            state.analysis_progress.set(None)
        # Cancellation-Banner löschen, falls die vorherige Session abgebrochen wurde.
        if hasattr(state, "analysis_cancelled"):
            state.analysis_cancelled.set(False)
        ui.update_text("name_group", value=config.get_parameters()['group_id'])
        ui.update_numeric("num_pupils", value=config.get_parameters()['num_pupils'])
        ui.update_text("name_teacher", value=config.get_parameters()['teacher_name'])

        # Close modal and go back to Analysis Pane
        ui.modal_remove()
        ui.update_navset("main_tabs", selected='<div id="loc_title_analysis" class="shiny-text-output"></div>')
