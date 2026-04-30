"""Analysis section: document input tab — transcript/codebook upload, format wizard."""
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
            ui.input_text("name_group", t("analysis", "group_id"), config.get_parameters()['group_id']),
            "tooltip_group_id",
        )

    @render.ui
    def loc_num_pupils():
        return _tt_wrap(
            ui.input_numeric("num_pupils", t("analysis", "num_pupils"), config.get_parameters()['num_pupils'], min=1, max=100),
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
                ui.input_file(
                    "transcript",
                    t("analysis", "upload_transcript"),
                    multiple=False,
                    accept=[".txt", ".docx"],
                    button_label=t("analysis", "browse"),
                    placeholder=t("analysis", "placeholder"),
                ),
                class_="ttai-file-wrap",
                style="flex: 0 1 auto; min-width: 0;",
                **{"data-tt-help": t("onboarding", "tooltip_upload_transcript")},
            ),
            ui.div(
                ui.output_ui("loc_transcript_format_status"),
                style="flex: 0 0 auto; align-self: flex-end; display: inline-flex; align-items: center;",
            ),
            ui.div(
                ui.tooltip(
                    ui.input_action_button(
                        "button_check_format",
                        "",
                        icon=icon_svg("wand-magic-sparkles"),
                        class_="btn-default",
                        # Quadratischer Button mit fester Größe (32x32 px),
                        # Icon mittig per inline-flex + line-height: 1.
                        style="width: 1.875rem; height: 1.875rem; padding: 0; display: inline-flex; align-items: center; justify-content: center; line-height: 1;",
                    ),
                    t("analysis", "check_format_tooltip"),
                    placement="right",
                ),
                # align-self: flex-end pinnt den Wand-Button an die Unterkante
                # des Wrap-Divs. Da wir die Progress-Bar (siehe CSS unten) und
                # den unteren Margin entfernt haben, endet das Wrap-Div exakt
                # an der Unterkante des Browse-Buttons — Wand sitzt damit
                # automatisch auf gleicher Höhe.
                style="flex: 0 0 auto; align-self: flex-end;",
            ),
            ui.tags.style(
                # margin-bottom + Progress-Bar-Reservierung raus, damit das
                # Wrap-Div exakt an der Unterkante des Browse-Buttons endet.
                # (Die Upload-Progress-Bar wird in dieser App nicht angezeigt;
                # ihr reservierter Block würde sonst die Höhe verfälschen und
                # je nach Upload-State springen lassen — siehe Bug-Report.)
                ".ttai-file-wrap .shiny-input-container,"
                ".ttai-file-wrap .form-group { margin-bottom: 0 !important; }"
                ".ttai-file-wrap .progress,"
                ".ttai-file-wrap .shiny-file-input-progress { display: none !important; }"
            ),
            style="display: flex; gap: 0.5rem; align-items: start;",
        )

    @render.ui
    def loc_transcript_format_status():
        return render_transcript_format_status_ui(
            state.transcript_format_status.get(), t
        )

    # Transkript verarbeiten
    @reactive.effect
    @reactive.event(input.transcript)
    def process_transcript():
        file = input.transcript()
        if file is not None:
            data = import_file(file[0])
            transcript_data.set(data)
            detected_teacher = detect_teacher_label(file[0])
            if detected_teacher:
                ui.update_text("name_teacher", value=detected_teacher)
                teacher = detected_teacher
            else:
                try:
                    teacher = input.name_teacher()
                except Exception:
                    teacher = None
            state.transcript_format_status.set(
                detect_transcript_format_status(file[0], teacher)
            )
        else:
            state.transcript_format_status.set(None)

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

    def _run_format_check(file):
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

    @reactive.effect
    @reactive.event(input.button_check_format)
    def check_transcript_format():
        _run_format_check(input.transcript())

    @reactive.effect
    @reactive.event(input.button_check_format_autopilot)
    def check_transcript_format_autopilot():
        try:
            file = input.autopilot_transcript()
        except Exception:
            file = None
        _run_format_check(file)

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
                accept=[".txt", ".docx"],
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
