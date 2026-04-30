from shiny import ui
from faicons import icon_svg


def build_autopilot_tab():
    """Tab layout for the Autopilot — one-click pipeline that codes a transcript
    twice (with two different LLMs) and pushes both runs into the Testing tab.

    Inputs (transcript + codebook) live on the same reactive values as the
    Analysis tab, so a user can move between tabs without re-uploading. The
    handler in ``handlers/autopilot.py`` fills every ``output_ui`` slot below.
    """
    return ui.nav_panel(
        ui.output_text("loc_title_autopilot"),
        ui.card(
            ui.card_header(ui.output_ui("loc_autopilot_intro")),
            ui.output_ui("loc_autopilot_intro_body"),
        ),
        ui.card(
            ui.card_header(ui.output_ui("loc_autopilot_inputs_header")),
            ui.layout_columns(
                ui.output_ui("loc_autopilot_upload_transcript"),
                ui.output_ui("loc_autopilot_upload_codebook"),
            ),
            ui.output_ui("loc_autopilot_inputs_status"),
        ),
        ui.card(
            ui.card_header(ui.output_ui("loc_autopilot_general_header")),
            ui.layout_columns(
                ui.output_ui("loc_autopilot_group_id"),
                ui.output_ui("loc_autopilot_num_pupils"),
                ui.output_ui("loc_autopilot_name_teacher"),
            ),
        ),
        ui.card(
            ui.card_header(ui.output_ui("loc_autopilot_options_header")),
            ui.output_ui("loc_autopilot_options"),
        ),
        ui.card(
            ui.card_header(ui.output_ui("loc_autopilot_coders_header")),
            ui.layout_columns(
                ui.output_ui("loc_autopilot_coder_a"),
                ui.output_ui("loc_autopilot_coder_b"),
            ),
            ui.output_ui("loc_autopilot_validation"),
        ),
        ui.card(
            ui.output_ui("loc_autopilot_reports_options"),
            ui.output_ui("loc_autopilot_start_button"),
            ui.output_ui("loc_autopilot_progress"),
        ),
        ui.output_ui("loc_autopilot_report_downloads"),
        ui.output_ui("loc_autopilot_results_section"),
        icon=icon_svg("plane-departure"),
    )
