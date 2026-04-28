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
