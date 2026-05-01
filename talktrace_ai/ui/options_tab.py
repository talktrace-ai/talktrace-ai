from shiny import ui
from faicons import icon_svg


def build_options_tab():
    return ui.nav_panel(
        ui.output_text("loc_title_options"),
        # API Configuration
        ui.card(
            ui.card_header(ui.output_ui("loc_api_configuration")),
            ui.layout_columns(
                # Select between OpenAI and Groq API
                ui.card(
                    ui.output_text("loc_api_select_title"),
                    ui.output_ui("loc_api_select"),
                ),
                # API Key Management
                ui.card(
                    ui.output_text("loc_api_key_exists"),
                    ui.layout_columns(
                        ui.output_ui("loc_button_change_api_key"),
                        ui.output_ui("loc_button_delete_api_key"),
                    ),
                ),
            ),
        ),
        # Manage models for LLM selection
        ui.card(
            ui.card_header(ui.output_ui("loc_llm_models")),
            ui.output_ui("loc_load_models"),
            ui.layout_columns(
                ui.output_ui("loc_button_add_model"),
                ui.output_ui("loc_button_remove_model"),
                ui.output_ui("loc_button_reset_model_selection"),
                col_widths=[3, 3],
            ),
        ),
        # Prompt Management for System and User Prompt
        ui.card(
            ui.card_header(ui.output_ui("loc_custom_prompts")),
            "System-Prompt",
            ui.output_text_verbatim("system_prompt_output"),
            ui.layout_columns(
                ui.output_ui("loc_button_change_system_prompt"),
                ui.output_ui("loc_button_reset_system_prompt"),
                col_widths=[2, 2],
            ),
            "User-Prompt",
            ui.output_text_verbatim("user_prompt_output"),
            ui.layout_columns(
                ui.output_ui("loc_button_change_user_prompt"),
                ui.output_ui("loc_button_reset_user_prompt"),
                col_widths=[2, 2],
            ),
        ),
        ui.card(
            ui.card_header(ui.output_ui("loc_additional_options")),
            ui.layout_columns(
                ui.output_ui("loc_input_teacher_name_options"),
                ui.output_ui("loc_input_group_id_options"),
                ui.output_ui("loc_input_num_pupils_options"),
                ui.output_ui("loc_button_reset_parameters"),
                col_widths=[2, 2, 2, 2],
            ),
        ),
        ui.card(
            ui.card_header(ui.output_ui("loc_advanced_options")),
            ui.output_ui("loc_streaming_switch"),
            ui.output_ui("loc_local_only_switch"),
        ),
        ui.card(
            ui.card_header(ui.output_ui("loc_cost_tracker_header")),
            ui.output_ui("cost_tracker_table"),
            ui.output_ui("loc_cost_tracker_reset_button"),
        ),
        ui.card(
            ui.card_header(ui.output_ui("loc_self_test_header")),
            ui.output_ui("loc_self_test_intro"),
            ui.output_ui("loc_self_test_button"),
            ui.output_ui("self_test_result"),
        ),
        icon=icon_svg("gear"),
    )
