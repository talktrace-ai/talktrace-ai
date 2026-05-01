"""Testing section: intercoder agreement / kappa analysis."""
from ._common import *


def register(state):
    input = state.input
    output = state.output
    session = state.session
    t = state.t
    report_a_df = state.report_a_df
    report_b_df = state.report_b_df
    report_a_error = state.report_a_error
    report_b_error = state.report_b_error

    ### Testen (Intercoder-Übereinstimmung) ------------------------------
    @render.ui
    def loc_title_testing():
        return tab_title_with_badge(
            t("testing", "tab_title"),
            state.tab_badge_testing.get(),
        )

    @reactive.effect
    @reactive.event(input.main_tabs)
    def _flip_testing_badge_on_visit():
        if main_tab_is(input.main_tabs(), "loc_title_testing"):
            if state.tab_badge_testing.get() == "unread":
                state.tab_badge_testing.set("read")

    @render.ui
    def loc_testing_header():
        return ui.p(t("testing", "section_header"))

    @render.ui
    def loc_testing_intro():
        return ui.p(t("testing", "intro"))

    def _glossary_tip(label_text, glossary_key):
        """Wrap a header label with a help-icon tooltip pulling from
        the localized glossary. Hover the icon to see the one-line
        definition + paper reference."""
        return ui.span(
            label_text,
            " ",
            ui.tooltip(
                ui.tags.span(
                    icon_svg("circle-info"),
                    class_="text-muted",
                    style="cursor: help; font-size: 0.85em;",
                ),
                t("glossary", glossary_key),
                placement="right",
            ),
        )

    @render.ui
    def loc_testing_kappa():
        return ui.p(_glossary_tip(t("testing", "kappa_header"), "kappa"))

    @render.ui
    def loc_testing_confusion():
        return ui.p(_glossary_tip(t("testing", "confusion_header"), "confusion_matrix"))

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
                ui.value_box(_glossary_tip(t("testing", "summary_n_pairs"), "n_pairs"),
                             str(res["n_pairs"]), theme="primary"),
                ui.value_box(_glossary_tip(t("testing", "summary_n_both"), "n_both"),
                             str(res["n_both"]), theme="success"),
                ui.value_box(_glossary_tip(t("testing", "summary_only_a"), "n_only"),
                             str(res["n_only_a"]), theme="warning"),
                ui.value_box(_glossary_tip(t("testing", "summary_only_b"), "n_only"),
                             str(res["n_only_b"]), theme="warning"),
            )
        )

        pa = res.get("percent_agreement", float("nan"))
        alpha = res.get("krippendorff_alpha", float("nan"))
        pa_str = f"{pa * 100:.1f} %" if pa == pa else "n/a"
        alpha_str = f"{alpha:.3f}" if alpha == alpha else "n/a"
        items.append(
            ui.layout_columns(
                ui.value_box(_glossary_tip(t("testing", "summary_percent_agreement"), "percent_agreement"),
                             pa_str, theme="info"),
                ui.value_box(_glossary_tip(t("testing", "summary_krippendorff"), "krippendorff_alpha"),
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
                _glossary_tip(f"κ = {k:.3f}", "kappa"),
                ui.tags.span(ci_text, style="font-size: 1.4rem; font-weight: 400; color: var(--bs-secondary-color); margin-left: 0.5rem;"),
                ui.tags.span(f" ({ci_caption})", style="font-size: 0.9rem; color: var(--bs-secondary-color);"),
                style="font-size: 2.4rem; font-weight: 600;",
            )
        else:
            value_html = ui.tags.div(
                _glossary_tip(f"κ = {k:.3f}", "kappa"),
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
            ui.tags.th(_glossary_tip(t("testing", "per_code_col_f1"), "f1")),
            ui.tags.th(_glossary_tip(t("testing", "per_code_col_precision"), "precision")),
            ui.tags.th(_glossary_tip(t("testing", "per_code_col_recall"), "recall")),
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

    def _current_testing_format():
        try:
            return input.testing_export_format() or "xlsx"
        except Exception:
            return "xlsx"

    def _testing_download_suffix(fmt):
        # CSV is delivered as a ZIP bundle; everything else mirrors the format.
        return ".zip" if fmt == "csv" else f".{fmt}"

    def _testing_export_labels():
        return {
            "title": t("testing", "export_title"),
            "sheet_overview": t("report_options", "sheet_overview"),
            "sheet_confusion": t("testing", "confusion_header"),
            "sheet_per_code": t("testing", "per_code_header"),
            "sheet_pairs": "Pairs",
        }

    @render.ui
    def testing_export_button():
        if _agreement() is None:
            return None
        return ui.div(
            ui.div(
                ui.input_select(
                    "testing_export_format",
                    t("report_options", "format_label"),
                    choices={
                        "xlsx": t("report_options", "format_xlsx"),
                        "csv": t("report_options", "format_csv"),
                        "json": t("report_options", "format_json"),
                        "html": t("report_options", "format_html"),
                        "docx": t("report_options", "format_docx"),
                        "pdf": t("report_options", "format_pdf"),
                    },
                    selected=_current_testing_format(),
                    width="220px",
                ),
                style="margin-bottom:0",
                class_="tt-testing-format",
            ),
            ui.download_button(
                "download_testing_report",
                t("testing", "export_report"),
                icon=icon_svg("download"),
                class_="btn-sm",
            ),
            ui.tags.style(
                ".tt-testing-format .shiny-input-container{margin-bottom:0!important}"
            ),
            style="display:flex;gap:0.75rem;align-items:flex-end;flex-wrap:wrap",
        )

    # ============== Expertenmodus =====================================
    expert_mode_on = state.expert_mode_on
    expert_metric = state.expert_metric
    expert_n_raters = state.expert_n_raters
    expert_result = state.expert_result
    expert_error = state.expert_error

    _METRIC_NAME_KEY = {
        "cohen": "expert_metric_name_cohen",
        "krippendorff": "expert_metric_name_krippendorff",
        "fleiss": "expert_metric_name_fleiss",
    }
    _METRIC_GLOSSARY_KEY = {
        "cohen": "kappa",
        "krippendorff": "krippendorff_alpha",
        "fleiss": "fleiss_kappa",
    }

    def _min_raters_for(metric):
        return 3 if metric == "fleiss" else 2

    @render.ui
    def loc_expert_mode_switch():
        return ui.div(
            ui.input_switch(
                "expert_mode_testing",
                t("testing", "expert_toggle"),
                value=expert_mode_on.get(),
            ),
            ui.tags.div(
                t("testing", "expert_toggle_hint"),
                class_="text-muted small",
                style="margin-top:-0.25rem;margin-bottom:0.5rem;",
            ),
            style="margin-top:0.5rem;",
        )

    def _show_expert_modal():
        ui.modal_show(ui.modal(
            ui.input_radio_buttons(
                "expert_metric_choice",
                t("testing", "expert_metric_label"),
                choices={
                    "cohen": t("testing", "expert_metric_cohen"),
                    "krippendorff": t("testing", "expert_metric_krippendorff"),
                    "fleiss": t("testing", "expert_metric_fleiss"),
                },
                selected=expert_metric.get(),
            ),
            ui.output_ui("loc_expert_n_raters_input"),
            ui.output_ui("loc_expert_file_inputs"),
            ui.output_ui("loc_expert_modal_error"),
            title=t("testing", "expert_modal_title"),
            easy_close=False,
            size="l",
            footer=(
                ui.input_action_button(
                    "expert_compute", t("testing", "expert_compute"),
                    class_="btn-success",
                ),
                ui.modal_button(
                    t("testing", "expert_cancel"), class_="btn-secondary",
                ),
            ),
        ))

    @reactive.effect
    @reactive.event(input.expert_mode_testing)
    def _on_expert_toggle():
        on = bool(input.expert_mode_testing())
        expert_mode_on.set(on)
        if on and expert_result.get() is None:
            expert_error.set(None)
            _show_expert_modal()

    @render.ui
    def loc_expert_n_raters_input():
        try:
            m = input.expert_metric_choice()
        except Exception:
            m = expert_metric.get()
        if m == "cohen":
            return None
        min_n = _min_raters_for(m)
        n_default = max(expert_n_raters.get(), min_n)
        return ui.input_numeric(
            "expert_n_raters_choice",
            t("testing", "expert_n_raters_label"),
            value=n_default,
            min=min_n,
            max=10,
            step=1,
        )

    def _resolved_n_raters():
        try:
            m = input.expert_metric_choice()
        except Exception:
            m = expert_metric.get()
        if m == "cohen":
            return 2, m
        try:
            raw = input.expert_n_raters_choice()
            n = int(raw) if raw not in (None, "") else expert_n_raters.get()
        except Exception:
            n = expert_n_raters.get()
        min_n = _min_raters_for(m)
        return max(min(int(n or min_n), 10), min_n), m

    @render.ui
    def loc_expert_file_inputs():
        n, _m = _resolved_n_raters()
        fields = []
        for i in range(1, n + 1):
            fields.append(ui.input_file(
                f"expert_file_{i}",
                t("testing", "expert_file_label_template").format(i=i),
                multiple=False,
                accept=[".docx", ".xlsx", ".html", ".htm"],
                button_label=t("analysis", "browse"),
                placeholder=t("testing", "placeholder_report"),
            ))
        return ui.div(*fields, style="margin-top:0.75rem;")

    @render.ui
    def loc_expert_modal_error():
        err = expert_error.get()
        if not err:
            return None
        return ui.tags.div(err, class_="text-danger",
                           style="margin-top:0.75rem;font-weight:500;")

    @reactive.effect
    @reactive.event(input.expert_compute)
    def _on_expert_compute():
        n, metric = _resolved_n_raters()
        if metric == "cohen" and n != 2:
            expert_error.set(t("testing", "expert_error_invalid_n_for_metric"))
            return
        if metric == "fleiss" and n < 3:
            expert_error.set(t("testing", "expert_error_invalid_n_for_metric"))
            return

        dfs = []
        for i in range(1, n + 1):
            try:
                f = input[f"expert_file_{i}"]()
            except Exception:
                f = None
            if not f:
                expert_error.set(t("testing", "expert_error_too_few_files"))
                return
            df, err = _parse_uploaded_report(f[0])
            if err:
                expert_error.set(f"Coder {i}: {err}")
                return
            dfs.append(df)

        expert_error.set(None)
        ui.notification_show(t("testing", "expert_computing"),
                             type="message", duration=2)
        try:
            res = compute_intercoder_agreement_multi(
                dfs, metric=metric,
                unmatched_label=t("testing", "unmatched_label"),
            )
        except ValueError:
            expert_error.set(t("testing", "expert_error_invalid_n_for_metric"))
            return
        except Exception:
            expert_error.set(t("testing", "expert_error_compute_failed"))
            return

        expert_metric.set(metric)
        expert_n_raters.set(n)
        expert_result.set(res)
        expert_mode_on.set(True)
        ui.modal_remove()

    @reactive.effect
    @reactive.event(input.expert_reconfigure)
    def _on_expert_reconfigure():
        expert_error.set(None)
        _show_expert_modal()

    @render.ui
    def loc_expert_mode_results():
        if not expert_mode_on.get():
            return None
        res = expert_result.get()
        if res is None:
            return None
        metric = res.get("metric", "cohen")
        metric_name = t("testing", _METRIC_NAME_KEY.get(metric, "expert_metric_name_cohen"))
        val = res.get("value", float("nan"))
        ci_low = res.get("ci_low", float("nan"))
        ci_high = res.get("ci_high", float("nan"))
        p = res.get("p_value", float("nan"))
        val_str = f"{val:.3f}" if val == val else "n/a"
        ci_str = (f"[{ci_low:.3f}, {ci_high:.3f}]"
                  if ci_low == ci_low and ci_high == ci_high else "n/a")
        p_str = f"{p:.4f}" if p == p else "n/a"
        stars = p_value_stars(p)
        if stars == "n.s.":
            stars_node = ui.tags.span(
                f" ({stars})",
                style="color:var(--bs-secondary-color);margin-left:0.4rem;font-size:0.95rem;font-weight:400;",
            )
        else:
            stars_node = ui.tags.span(
                f" {stars}",
                style="color:var(--bs-success);margin-left:0.4rem;font-weight:600;",
            )
        glossary_key = _METRIC_GLOSSARY_KEY.get(metric, "kappa")
        return ui.card(
            ui.card_header(t("testing", "expert_results_header")),
            ui.tags.div(
                ui.tags.div(
                    _glossary_tip(f"{metric_name} = {val_str}", glossary_key),
                    stars_node,
                    style="font-size:2.2rem;font-weight:600;line-height:1.2;",
                ),
                ui.tags.div(
                    _glossary_tip(f"{t('testing', 'expert_result_ci')}: {ci_str}", "ci"),
                    style="color:var(--bs-secondary-color);margin-top:0.25rem;",
                ),
                ui.tags.div(
                    _glossary_tip(f"{t('testing', 'expert_result_p_value')}: {p_str}", "p_value"),
                    style="color:var(--bs-secondary-color);",
                ),
                ui.layout_columns(
                    ui.value_box(
                        t("testing", "expert_result_n_units"),
                        str(res.get("n_units", "")),
                        theme="primary",
                    ),
                    ui.value_box(
                        t("testing", "expert_result_n_raters"),
                        str(res.get("n_raters", "")),
                        theme="success",
                    ),
                    col_widths=[6, 6],
                ),
                ui.input_action_button(
                    "expert_reconfigure",
                    t("testing", "expert_reconfigure"),
                    icon=icon_svg("gear"),
                    class_="btn-sm btn-outline-secondary",
                ),
                style="padding:0.5rem 0.25rem;",
            ),
        )

    # ============== /Expertenmodus ====================================

    @render.download(
        filename=lambda: f"{date.today().isoformat()} - Intercoder Agreement{_testing_download_suffix(_current_testing_format())}"
    )
    def download_testing_report():
        res = _agreement()
        if res is None:
            ui.notification_show(t("testing", "no_data"), type="warning", duration=4)
            return None
        fmt = _current_testing_format()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=_testing_download_suffix(fmt))
        tmp_file.close()
        try:
            export_testing_agreement_any(
                tmp_file.name, res, fmt, labels=_testing_export_labels(),
            )
        except RuntimeError as e:
            key = str(e)
            known = {"xlsx_unavailable", "docx_unavailable",
                     "pdf_unavailable", "pdf_unavailable_linux"}
            msg = t("report_options", key) if key in known else str(e)
            ui.notification_show(msg, type="error", duration=6)
            return None
        return tmp_file.name
