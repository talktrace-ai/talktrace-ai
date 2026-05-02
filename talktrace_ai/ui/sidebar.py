from shiny import ui
from faicons import icon_svg


def build_sidebar():
    return ui.sidebar(
        ui.input_action_button("language_toggle", "English", icon=icon_svg("globe")),
        ui.output_ui("loc_dynamic_model_select"),
        ui.output_ui("loc_provider_hint"),
        ui.output_ui("loc_llm_switch"),
        ui.output_ui("loc_analyse_speakers_switches"),
        ui.output_ui("cost_chip"),
        ui.output_ui("loc_button_analysis"),
        ui.output_ui("loc_button_cancel_analysis"),
        ui.output_ui("start_analysis"),
        ui.output_ui("show_report_download_button"),
        ui.output_ui("loc_button_import_session"),
        ui.output_ui("loc_button_export_session"),
        ui.output_ui("loc_button_history"),
        ui.output_ui("loc_button_reset"),
    )
