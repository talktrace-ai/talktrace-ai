from shiny import ui
from faicons import icon_svg


def build_testing_tab():
    return ui.nav_panel(
        ui.output_text("loc_title_testing"),
        ui.card(
            ui.card_header(ui.output_ui("loc_testing_header")),
            ui.output_ui("loc_testing_intro"),
            ui.layout_columns(
                ui.output_ui("loc_upload_report_a"),
                ui.output_ui("loc_upload_report_b"),
            ),
            ui.output_ui("testing_summary"),
            ui.output_ui("testing_export_button"),
        ),
        ui.card(
            ui.card_header(ui.output_ui("loc_testing_kappa")),
            ui.output_ui("testing_kappa_value"),
        ),
        ui.card(
            ui.card_header(ui.output_ui("loc_testing_confusion")),
            ui.output_ui("testing_confusion_table"),
            full_screen=True,
        ),
        ui.card(
            ui.card_header(ui.output_ui("testing_per_code_header")),
            ui.output_ui("testing_per_code_table"),
            full_screen=True,
        ),
        icon=icon_svg("scale-balanced"),
    )
