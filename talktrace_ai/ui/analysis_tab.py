from shiny import ui
from faicons import icon_svg


def build_analysis_tab():
    return ui.nav_panel(
        ui.output_text("loc_title_analysis"),
        ui.card(
            ui.layout_columns(
                # Group and Transcript Metadata
                ui.card(
                    ui.card_header(ui.output_ui("loc_general_info")),
                    ui.output_ui("loc_group_id"),
                    ui.output_ui("loc_num_pupils"),
                    ui.output_ui("loc_name_teacher"),
                ),
                # Document Upload for Transcript and Codebook
                ui.card(
                    ui.card_header(ui.output_ui("loc_document_input")),
                    ui.output_ui("loc_upload_transcript"),
                    ui.output_ui("loc_upload_codebook"),
                ),
            ),
        ),
        # Preview of Codebook and Transcript
        ui.card(
            ui.card_header(ui.output_ui("loc_preview_codebook")),
            ui.output_ui("show_codebook_preview"),
            full_screen=True,
        ),
        ui.card(
            ui.card_header(ui.output_ui("loc_general_transcript")),
            ui.output_ui("show_transcript_preview"),
            full_screen=True,
        ),
        icon=icon_svg("brain"),
    )
