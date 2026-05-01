from shiny import ui
from faicons import icon_svg


def build_info_tab():
    return ui.nav_panel(
        ui.output_text("loc_title_info"),
        ui.tags.a(
            ui.tags.img(
                src="/tt-assets/banner-light.png",
                alt="TalkTrace AI neo",
                style="max-width:100%;height:auto;display:block;margin:0 auto;",
            ),
            href="https://github.com/MoominVibeCoder/talktrace-ai-neo",
            target="_blank",
            rel="noopener noreferrer",
            style="display:block;margin-bottom:1.25rem;",
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.output_ui("loc_info_dev_heading")),
                ui.output_ui("loc_info_dev_body"),
            ),
            ui.card(
                ui.card_header(ui.output_ui("loc_info_license_heading")),
                ui.output_ui("loc_info_license_body"),
            ),
            col_widths=[6, 6],
        ),
        icon=icon_svg("circle-info"),
    )
