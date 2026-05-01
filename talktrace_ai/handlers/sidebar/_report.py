"""Report download: button, format/sections modal, download handler."""
from .._common import *


def register(state):
    input = state.input
    t = state.t
    config = state.config
    transcript_data = state.transcript_data
    num_participants = state.num_participants
    participation_rate = state.participation_rate
    t_turns = state.t_turns
    t_turns_length = state.t_turns_length
    t_turns_length_mean_sd = state.t_turns_length_mean_sd
    p_turns = state.p_turns
    p_turns_length = state.p_turns_length
    p_turns_length_mean_sd = state.p_turns_length_mean_sd
    llm_analysis_data = state.llm_analysis_data
    model = state.model
    teacher_impulses_count = state.teacher_impulses_count
    analysis_state = state.analysis_state
    analysis_llm_state = state.analysis_llm_state
    sim_plot = state.sim_plot
    qual_plot = state.qual_plot
    qual_stats_df = state.qual_stats_df
    code_legend_storage = state.code_legend_storage

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
                    "csv": t("report_options", "format_csv"),
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


    def _report_file_suffix(fmt):
        # CSV is delivered as a ZIP bundle; everything else mirrors the format.
        return ".zip" if fmt == "csv" else f".{fmt}"

    @render.download(filename=lambda: f"{date.today().isoformat()} - TalkTrace AI {t('results', 'results_group')} {input.name_group()}{_report_file_suffix(_current_report_format())}")
    def download_report():
        sections = _current_report_sections()
        fmt = _current_report_format()
        # Persist last selection for the next modal open.
        report_options.set({"sections": dict(sections), "format": fmt})

        if not any(sections.values()):
            ui.notification_show(t("report_options", "no_section_selected"), type="warning", duration=4)
            return None

        suffix = _report_file_suffix(fmt)
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

        # Reproducibility fingerprint: pins down codebook + prompts + model +
        # transcript so reviewers can verify the run was produced from the
        # exact configuration recorded in the report legend.
        try:
            fp = compute_fingerprint(
                state.codebook_data.get(),
                state.effective_system_prompt(),
                state.effective_user_prompt(),
                model.get() or "",
                transcript_data.get(),
            )
        except Exception:
            fp = ""

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
                fingerprint=fp,
            )
        except RuntimeError as e:
            key = str(e)
            msg = t("report_options", key) if key in ("pdf_unavailable", "pdf_unavailable_linux", "xlsx_unavailable") else str(e)
            ui.notification_show(msg, type="error", duration=6)
            return None

        ui.modal_remove()
        return tmp_file.name
