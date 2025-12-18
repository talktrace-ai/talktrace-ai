import re
from httpx import get
from matplotlib.style import available
from numpy import extract, place
from .myfuncs import generate_report2, import_file, count_pupils, dialog_stats, count_teacher_impulses, llm_analysis_groq, llm_analysis_openai
from .config.config_manager import ConfigManager
from .localization.translation import TRANSLATIONS

from pathlib import Path
import sys
import os
import webbrowser
from shiny import App, render, ui, reactive, req
from shiny._main import run_app

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from faicons import icon_svg
from groq import Groq
from openai import OpenAI, api_key, models
import json
from datetime import date
import tempfile
import pickle
import keyring
import keyring.errors
import tiktoken


# Path Helper for css-files
def resource_path(relative_path: str) -> Path:
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).parent / relative_path

# Open the App in a Web Browser
url = "http://127.0.0.1:8000"
webbrowser.open_new_tab(url)

# Define the Layout

# Sidebar Menu with Model Selection, Analysis, Report Download and Session Management
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_action_button("language_toggle", "English", icon=icon_svg("globe")),        
        ui.output_ui("loc_dynamic_model_select"),
        ui.output_ui("loc_llm_switch"),
        ui.output_ui("loc_display_cost_prediction"),
        ui.output_ui("loc_button_analysis"),
        ui.output_text("start_analysis"),
        ui.output_ui("show_report_download_button"),
        ui.output_ui("loc_button_import_session"),
        ui.output_ui("loc_button_export_session"),
        ui.output_ui("loc_button_reset"),
        #title="Controls",
    ),

    # Main Content Area with Tabs for Analysis, Results, and Options
    ui.navset_tab(  
        ui.nav_panel(ui.output_text("loc_title_analysis"),
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
            icon=icon_svg("brain")
        ),
        # Results Tab with Quantitative and Qualitative Analysis
        ui.nav_panel(ui.output_text("loc_title_results"),
            ui.card(
                ui.card_header(ui.output_ui("loc_quantitative_analysis")),
                # General group satistics 
                ui.layout_column_wrap(
                    ui.output_ui("loc_group_id_display"),
                    ui.output_ui("loc_class_size"),
                    ui.output_ui("loc_num_participants"),
                    ui.output_ui("loc_participation_rate"),                   
                    fill=False,
                ),
                ),
                # Quantitative Stats for Conversation Distribution
                ui.layout_columns(
                    # Conversation Distribution Plot
                    ui.card(
                        ui.card_header(ui.output_ui("loc_distribution_of_turns")),
                        ui.output_plot("sim_stats_plot"),
                        full_screen=True,
                    ),
                    # Conversation Statistics for Teacher and Pupils
                    ui.card(
                        ui.card_header(ui.output_ui("loc_interaction_turns")),
                        ui.layout_column_wrap(
                            ui.card(
                            ui.card_header(ui.output_ui("loc_teacher")),
                            ui.output_ui("loc_teacher_turns"),
                            ui.output_text("teacher_turns"),
                            ui.output_ui("loc_teacher_turns_length"),
                            ui.output_text("teacher_turns_length"),

                            ),
                            ui.card(
                            ui.card_header(ui.output_ui("loc_pupils")),
                            ui.output_ui("loc_pupils_turns"),
                            ui.output_text("pupils_turns"),
                            ui.output_ui("loc_pupils_turns_length"),
                            ui.output_text("pupils_turns_length"),
                            ),
                        ),
                        full_screen=True,
                    ),
                    col_widths=[4, 8]
                ),
            # Qualitative Analysis of Teacher's Impulses
            # Quick Stats for Teacher's Impulses
            ui.card(
                ui.card_header(ui.output_ui("loc_qualitative_analysis")),
                ui.layout_columns(
                    ui.value_box(
                        ui.output_ui("loc_impulses_count"),
                        ui.output_text("teacher_impulses"),
                        showcase=icon_svg("square-poll-vertical")
                    ),
                    ui.value_box(
                        ui.output_ui("loc_coded_impulses"),
                        ui.output_text("teacher_impulses_coded"),
                        showcase=icon_svg("hashtag")
                    ),
                    ui.value_box(
                        ui.output_ui("loc_most_frequent_codes"),
                        ui.output_text("code_most_used"),
                        showcase=icon_svg("ranking-star")
                    ),
                    ui.value_box(
                        ui.output_ui("loc_teacher_talking_rate"),
                        ui.output_text("teacher_share"),
                        showcase=icon_svg("user-tie")
                    ),
                    col_widths=[3]
                ),   
                ui.layout_columns(
                    # Qualitative Statistics Plot for Coded Impulses
                    ui.card(
                        ui.card_header(ui.output_ui("loc_impulses_distribution")),
                        ui.row(
                            ui.output_plot("qualitative_stats_plot"),
                        ),
                        # Explanation of Codes
                        ui.row(
                            ui.output_ui("code_legend"),
                        ),
                        full_screen=True,
                    ),
                    # DataFrame of Coded Impulses
                    ui.card(
                        ui.card_header(ui.output_ui("loc_impulses_coding")),
                        ui.output_ui("quali_stats_df"),
                        full_screen=True,
                    ),
                ),          
            ),
            icon=icon_svg("chart-bar")
        ),
        # Options Tab for API Configuration and Custom Prompts
        ui.nav_panel(ui.output_text("loc_title_options"),
            # API Configuration 
            ui.card(
                ui.card_header(ui.output_ui("loc_api_configuration")),
                ui.layout_columns(
                    # Select between OpenAI and Groq API    
                    ui.card(
                        ui.output_text("loc_api_select_title"),
                        ui.output_ui("loc_api_select")
                    ),
                    # Api Key Management
                    ui.card(
                        ui.output_text("loc_api_key_exists"),
                        ui.layout_columns( 
                        ui.output_ui("loc_button_change_api_key"),
                        ui.output_ui("loc_button_delete_api_key"),
                        ),
                    ),
                ),
            ),
            # Modelle für LLM-Auswahl verwalten
            ui.card(
                ui.card_header(ui.output_ui("loc_llm_models")),
                ui.output_ui("loc_load_models"),
                ui.layout_columns(
                    ui.output_ui("loc_button_add_model"),
                    ui.output_ui("loc_button_remove_model"),
                    ui.output_ui("loc_button_reset_model_selection"),
                    col_widths=[3,3]
                )

            ),
                # Prompt Management for System and User Prompt
            ui.card(
                ui.card_header(ui.output_ui("loc_custom_prompts")),
                "System-Prompt",
                ui.output_text_verbatim("system_prompt_output"),
                ui.layout_columns(
                    ui.output_ui("loc_button_change_system_prompt"),
                    ui.output_ui("loc_button_reset_system_prompt"),
                    col_widths=[2,2]
                ),
                "User-Prompt",
                ui.output_text_verbatim("user_prompt_output"),
                ui.layout_columns(
                    ui.output_ui("loc_button_change_user_prompt"),
                    ui.output_ui("loc_button_reset_user_prompt"),
                    col_widths=[2,2]
                ),                                            
            ),
            ui.card(
                ui.card_header(ui.output_ui("loc_additional_options")),
                ui.layout_columns(
                    ui.output_ui("loc_input_teacher_name_options"),
                    ui.output_ui("loc_input_group_id_options"),
                    ui.output_ui("loc_input_num_pupils_options"),
                    ui.output_ui("loc_button_reset_parameters"),
                    col_widths=[2,2,2,2]
                ),
            ),
            ui.card(
                ui.card_header(ui.output_ui("loc_app_info")),
                ui.output_ui("loc_app_info_text"),
            ),
            icon=icon_svg("gear"),
        ), 
        # Tab Identifier to Actively Switch Between Tabs
        id="main_tabs",  
    ),  

    # Incluse CSS Stylesheet
    ui.include_css(str(resource_path("static/styles.css"))),
    # Set the Title of the App-Window
    title="TalkTrace AI",
    fillable=True
)


def server(input, output, session):

    # Initialize Config Manager to handle config file
    config = ConfigManager() 
    # Define helper variables
    transcript_data = reactive.value(None)
    codebook_data = reactive.value(None)
    api_key_groq = reactive.value()
    api_key_openai = reactive.value()
    current_api = reactive.value(config.get_current_api())
    num_participants = reactive.value(None)
    participation_rate = reactive.value(None)
    t_turns = reactive.value(None)
    t_turns_length = reactive.value(None)
    t_turns_length_mean_sd = reactive.value(None)
    p_turns = reactive.value(None)
    p_turns_length = reactive.value(None)
    p_turns_length_mean_sd = reactive.value(None)
    stats = reactive.value(None)
    llm_analysis_data = reactive.value([])
    model = reactive.value(config.get_current_model())
    teacher_impulses_count = reactive.value(None)
    analysis_state = reactive.value(False)
    analysis_llm_state = reactive.value(False)
    sim_plot = reactive.value(None)
    qual_plot = reactive.value()
    qual_stats_df = reactive.value(None)
    placeholder_plot = reactive.value()
    model_deleted = reactive.value(0) # for reactivitiy of model selection after model deletion
    current_lang = reactive.value(config.get_localization()["current_language"])
    code_legend_storage = reactive.value("Legende nicht ausgelesen")
    estimated_cost = reactive.value(None)
    token_count = reactive.value(None)

    ### Localization
    # Helper function to get translated text
    def t(section, key):
        return TRANSLATIONS[current_lang.get()][section][key] 

    # Update language based on user selection
    @reactive.effect
    @reactive.event(input.language_toggle)
    def _():
        req(input.language_toggle())
        # Toggle between languages
        new_lang = "en" if current_lang.get() == "de" else "de"
        # Update button text and icon
        if new_lang == "en":
            ui.update_action_button(
                "language_toggle",
                label="Deutsch",
                icon=icon_svg("globe")
            )
            config.set_localization('current_language', 'en')
        else:
            ui.update_action_button(
                "language_toggle",
                label="English",
                icon=icon_svg("globe")
            )
            config.set_localization('current_language', 'de')

        current_lang.set("de" if new_lang == "de" else "en")


    # Define Baseline System Prompt
    system_prompt = reactive.value(config.get_prompts()['system'])

    # Define Baseline User Prompt
    user_prompt = reactive.value(config.get_prompts()['user'])

    # Fill Fields from Config File
    ui.update_text("name_group", value=config.get_parameters()['group_id'])
    ui.update_numeric("num_pupils", value=config.get_parameters()['num_pupils'])
    ui.update_text("name_teacher", value=config.get_parameters()['teacher_name'])
    ui.update_text("name_teacher_options", value=config.get_parameters()['teacher_name'])
    ui.update_text("name_group_options", value=config.get_parameters()['group_id'])
    ui.update_numeric("num_pupils_options", value=config.get_parameters()['num_pupils'])
    ui.update_action_button("language_toggle", icon=icon_svg("globe"), label="English" if config.get_localization()['current_language'] == 'de' else "Deutsch")


    try:
        api_key_openai.set(keyring.get_password("talktrace", "api_key_openai"))
    except keyring.errors.PasswordDeleteError:
        pass

    try: 
        api_key_groq.set(keyring.get_password("talktrace", "api_key_groq"))
    except keyring.errors.PasswordDeleteError:
        pass



    ### Sidebar --------------------------------------------------------
    # Model Selection
    @render.ui
    def loc_dynamic_model_select():
        return ui.input_select("model_select", t("sidebar", "model_select"), choices=select_api_choices(), selected=config.get_current_model())

    
    @reactive.effect()
    def update_current_model():
        model.set(input.model_select())
        config.set_current_model(input.model_select())
    

    # LLM Analyse
    @render.ui
    def loc_llm_switch():
        return ui.input_switch("llm_switch", t("sidebar", "llm_switch"), True)


    def calculate_input_tokens(transcript, codebook, system_prompt_text, user_prompt_text):
        """Calculate approximate token count for LLM request"""
        try:
            # Use the encoding for the selected model
            if config.get_current_api() == "openai":
                try: 
                    encoding = tiktoken.encoding_for_model(model.get())
                except:
                    encoding = tiktoken.get_encoding("cl100k_base")
            else:  # groq
                encoding = tiktoken.get_encoding("cl100k_base")
            
            # Combine all text
            all_text = f"{system_prompt_text}\n{user_prompt_text}\n{str(transcript)}\n{str(codebook)}"
            
            # Count tokens
            tokens = len(encoding.encode(all_text))
            return tokens
        except Exception as e:
            print(f"Token calculation error: {e}")
            return 0


    def calculate_estimated_cost(tokens):
        """Calculate estimated cost based on token count and selected API/model"""
        pricing = config.get_api_pricing()  # Add this to ConfigManager
        api = config.get_current_api()
        current_model = model.get()
        
        if api in pricing and current_model in pricing[api]:
            rate_in = pricing[api][current_model]["input"]  # Cost per 1K tokens
            rate_out = pricing[api][current_model]["output"]
            cost = (tokens / 1000000) * rate_in + (tokens / 1000000) * rate_out * 4
            return cost
        return None


    # Update cost prediction when transcript/codebook changes
    @reactive.effect
    def update_cost_prediction():
        req(transcript_data.get() != None, codebook_data.get() != None, input.llm_switch())
        tokens = calculate_input_tokens(
            transcript_data.get(),
            codebook_data.get() or "",
            system_prompt.get(),
            user_prompt.get()
        )
        token_count.set(tokens)
        cost = calculate_estimated_cost(tokens)
        estimated_cost.set(cost)


    @render.text
    def loc_display_cost_prediction():
        req(transcript_data.get() != None, codebook_data.get() != None)
        if input.llm_switch():
            tokens = token_count.get()
            cost = estimated_cost.get()
            if tokens and cost:
                return f"{t("sidebar", "tokens_aprox")} {tokens:} {t("sidebar", "cost_prediction")}: {cost:.4f} €"
        return ""


    # Start Analysis Button
    @render.ui
    def loc_button_analysis():
        return ui.input_action_button("button_analysis", t("sidebar", "button_analysis"), icon=icon_svg("magnifying-glass-chart"), class_="btn-success")

    # Shared analysis function
    async def run_analysis():
        req(transcript_data.get() != None)
        # Progress bar to indicate the analysis steps
        with ui.Progress(min=1, max=4) as p:
            p.set(message=t("system_prompts", "analysis_running"), detail=t("system_prompts", "wait"))
            # Generate Quantitative Stats without LLM 
            num_participants.set(count_pupils(transcript_data.get())) 
            p.set(1, message=t("system_prompts", "calculating"))
            
            stats.set(dialog_stats(transcript_data.get(), input.name_teacher()))
            teacher_impulses_count.set(count_teacher_impulses(stats.get(), input.name_teacher()))
            p.set(2, message=t("system_prompts", "waiting_LLM"))

            # Perform LLM-Request, if Activated
            if input.llm_switch():
                req(input.codebook())
                # Call either Groq or OpenAI API based on User Selection
                if config.get_current_api() == "groq":
                    req(api_key_groq.get() != None)
                    llm_response = llm_analysis_groq(system_prompt.get(), user_prompt.get(), model.get(), transcript_data.get(), codebook_data.get(), Groq(api_key=api_key_groq.get()))
                elif config.get_current_api() == "openai":
                    req(api_key_openai.get() != None)
                    llm_response = llm_analysis_openai(system_prompt.get(), user_prompt.get(), model.get(), transcript_data.get(), codebook_data.get(), OpenAI(api_key=api_key_openai.get()))
                
                if '"error":' in llm_response:
                    return f"❗️{t("system_prompts", "error")}: {json.loads(llm_response)['error']}. {t("system_prompts", "try_again")}"
                
                existing_data = llm_analysis_data.get()
                new_data = json.loads(llm_response)
                new_data_df = pd.DataFrame(new_data['analysis'], columns=['#', "Shortcode","Impuls"])

                existing_data.append(new_data_df)
                llm_analysis_data.set(list(existing_data)) # Important to Set as a List to Avoid Reactivity Issues, Due to Immutability Logic of Python!!!
                analysis_llm_state.set(True)
            p.set(4, message=t("sidebar", "analysis_completed"))
            # Mark Analysis as Completed
            analysis_state.set(True)
        # Automatically Switch to Results Tab
        ui.update_navs("main_tabs", selected='<div id="loc_title_results" class="shiny-text-output"></div>')
        return t("sidebar", "analysis_completed")

    # Analyse starten
    @render.text
    @reactive.event(input.button_analysis)
    async def start_analysis():
        return await run_analysis()

    @render.ui
    def show_report_download_button():
        req(analysis_state.get())
        return ui.download_button("download_report", t("sidebar", "download_report"), icon = icon_svg("download")),


    @render.download(filename=lambda: f"{date.today().isoformat()} - TalkTrace AI {t("results", "results_group")} {input.name_group.get()}.docx")
    def download_report():
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        tmp_file.close()
        if llm_analysis_data.get():
            generate_report2(tmp_file.name, input.name_group(), input.num_pupils(), num_participants.get(), participation_rate.get(), {"num": t_turns.get(), "words": t_turns_length.get(), "mean_sd": t_turns_length_mean_sd.get()}, {"num": p_turns.get(), "words": p_turns_length.get(), "mean_sd": p_turns_length_mean_sd.get()}, sim_plot.get(), teacher_impulses_count.get(), code_legend_storage.get(), True, qual_plot.get(), qual_stats_df.get())

        else:
            generate_report2(tmp_file.name, input.name_group(), input.num_pupils(), num_participants.get(), participation_rate.get(), {"num": t_turns.get(), "words": t_turns_length.get(), "mean_sd": t_turns_length_mean_sd.get()}, {"num": p_turns.get(), "words": p_turns_length.get(), "mean_sd": p_turns_length_mean_sd.get()}, sim_plot.get(), teacher_impulses_count.get(), llm_analysis=False, caption=code_legend_storage.get())

        return tmp_file.name


    # Import Session
    @render.ui
    def loc_button_import_session():
        return ui.input_file("button_import_session", t("sidebar", "import_session"), accept=[".pkl"], multiple=False, placeholder=t("analysis", "placeholder"), button_label=t("analysis", "browse")),


    @reactive.effect
  #  @reactive.event(input.button_import_session)
    async def button_import_session():
        
        file = input.button_import_session()

        if not file:
            return
    
        with open(file[0]["datapath"], "rb") as f:
            session_data = pickle.load(f)
        
        # Set the reactive values
        with reactive.isolate(): 
            try:
                transcript_data.set(session_data.get("transcript_data"))
                num_participants.set(session_data.get("num_participants"))
                participation_rate.set(session_data.get("participation_rate"))
                stats.set(session_data.get("stats"))
                llm_analysis_data.set(session_data.get("llm_analysis_data"))
                analysis_llm_state.set(session_data.get("analysis_llm_state"))
                placeholder_plot.set(session_data.get("placeholder_plot"))
                code_legend_storage.set(session_data.get("code_legend_storage"))
                ui.update_switch("llm_switch", value=False)
            except Exception as e:
                pass

        await run_analysis()
        '''
        m = ui.modal(  
                t("analysis", "modal_restart_analysis"),  
                title=t("analysis", "modal_title_attention"), 
                easy_close=True,
                footer=ui.modal_button("OK", class_="btn-success")  
            )  
        ui.modal_show(m)  
        '''

    # Export Session
    @render.ui
    def loc_button_export_session():
        return ui.download_button("button_export_session", t("sidebar", "export_session"), icon = icon_svg("file-export")),

    
    @render.download(filename=lambda: f"{date.today().isoformat()} - TalkTrace AI Session - {t("results", "results_group")} {input.name_group()} - {config.get_current_model}.pkl")
    def button_export_session():
        session_data = {
            "transcript_data": transcript_data.get(),
            "num_participants": num_participants.get(),
            "participation_rate": participation_rate.get(),
            "stats": stats.get(),
            "llm_analysis_data": llm_analysis_data.get(),
            "analysis_llm_state": analysis_llm_state.get(),
            "code_legend_storage": code_legend_storage.get(),
        }
        
        # serialize the dictionary to a pickle file
        with open("session_dump.pkl", "wb") as f:
            pickle.dump(session_data, f)
        return "session_dump.pkl"


    # Reset Session
    @render.ui
    def loc_button_reset():
        return ui.input_action_button("button_reset", t("sidebar", "reset_session"), icon = icon_svg("arrow-rotate-left"), class_="btn-danger"),

    @reactive.effect
    @reactive.event(input.button_reset)
    def reset_session():
        m = ui.modal(
            t("analysis", "modal_reset_session"),
            title=t("analysis", "modal_title_reset"),
            easy_close=True,
            footer=(ui.input_action_button("button_confirm_session_reset", t("analysis", "modal_confirm_reset"), class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"), class_="btn-danger")),
        )
        ui.modal_show(m)

    # Reset all reactive values to their initial state
    @reactive.effect
    @reactive.event(input.button_confirm_session_reset)
    def confirm_reset_session():
        transcript_data.set(None)
        codebook_data.set(None)
        num_participants.set(None)
        participation_rate.set(None)
        stats.set(None)
        llm_analysis_data.set([])
        teacher_impulses_count.set(None)
        analysis_state.set(False)
        analysis_llm_state.set(False)
        sim_plot.set(None)
        qual_plot.set(None)
        qual_stats_df.set(None)
        placeholder_plot.set(None)
        code_legend_storage.set("Legende nicht ausgelesen")
        ui.update_text("name_group", value=config.get_parameters()['group_id'])
        ui.update_numeric("num_pupils", value=config.get_parameters()['num_pupils'])
        ui.update_text("name_teacher", value=config.get_parameters()['teacher_name'])

        # Close modal and go back to Analysis Pane
        ui.modal_remove()
        ui.update_navs("main_tabs", selected='<div id="loc_title_analysis" class="shiny-text-output"></div>')

    ### Analyse --------------------------------------------------------

    @render.text
    def loc_title_analysis():
        return (t("analysis", "tab_title"))

    # Allgemeine Informationen
    @render.ui
    def loc_general_info():
        return ui.p(t("analysis", "general_info"))
    
    @render.ui
    def loc_group_id():
        return ui.input_text("name_group", t("analysis", "group_id"), "B1")
    
    @render.ui
    def loc_num_pupils():
        return ui.input_numeric("num_pupils", t("analysis", "num_pupils"), 25, min=1, max=100)

    @render.ui
    def loc_name_teacher():
        return ui.input_text("name_teacher", t("analysis", "name_teacher"), config.get_parameters()['teacher_name'])
    
    # Dokumenteneingabe
    @render.ui
    def loc_document_input():
        return ui.p(t("analysis", "document_input"))

    # Transkript Upload
    @render.ui
    def loc_upload_transcript():
        return ui.input_file(
            "transcript",
            t("analysis", "upload_transcript"),
            multiple=False,
            accept=[".txt", ".docx", ".pdf"],
            button_label=t("analysis", "browse"),
            placeholder=t("analysis", "placeholder"),
        )

    # Transkript verarbeiten
    @reactive.effect
    @reactive.event(input.transcript)
    def process_transcript():
        file = input.transcript()
        if file is not None:
            transcript_data.set(import_file(file[0]))
    

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
        return ui.input_file(
            "codebook",
            t("analysis", "upload_codebook"),
            multiple=False,
            accept=[".txt", ".docx", ".pdf"],
            button_label=t("analysis", "browse"),
            placeholder=t("analysis", "placeholder"),
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
        return ui.p(t("analysis", "preview_codebook"))


    @render.ui
    def show_codebook_preview():
        if codebook_data.get() == None:
            return t("analysis", "placeholder_codebook")
        else: 
            return ui.output_table("codebook_preview")
              

    @render.table
    def codebook_preview():
        req(codebook_data.get() != None)
        return pd.DataFrame(codebook_data.get()).iloc[1:] # Erste Zeile entfernen, da sie nur die Überschriften enthält


    # Vorschau Transkript
    @render.ui
    def loc_general_transcript():
        return ui.p(t("analysis", "preview_transcript"))


    @render.ui
    def show_transcript_preview():
        if transcript_data.get() == None:
            return t("analysis", "placeholder_transcript")
        else: 
            return transcript_data.get()


    ### Ergebnisse --------------------------------------------------------
    # Ergebnisse Tab Titel
    @render.text
    def loc_title_results():
        return (t("results", "tab_title"))
    

    # Warnung, wenn Ergebnisse Tab ohne Analyse angeklickt wird
    @reactive.effect
    @reactive.event(input.main_tabs)
    def warn_if_results_tab_clicked():
        if input.main_tabs() == '<div id="loc_title_results" class="shiny-text-output"></div>' and not analysis_state.get():
            m = ui.modal(
                ui.p(t("results", "no_results")),
                title=t("results", "no_results_title"),
                easy_close=True,
                footer=ui.modal_button("OK",  class_="btn-success"),
                size="m"
            )
            ui.modal_show(m)
            ui.update_navs("main_tabs", selected='<div id="loc_title_analysis" class="shiny-text-output"></div>')

    # Anzeige der allgemeinen Informationen
    @render.ui
    def loc_quantitative_analysis():
        return ui.h3(t("results", "section_quantitative_analysis"))


     # Update Quantiative Statistics Values
    @reactive.effect
    @reactive.event(input.button_analysis)
    def stats_values():
        req(transcript_data.get() != None)
        t_turns.set(stats.get().loc[stats.get()['Sprecher'] == input.name_teacher(), 'Anzahl_Beitraege'].values[0])
        t_turns_length.set(round(stats.get().loc[stats.get()['Sprecher'] == input.name_teacher(), 'Durchschnitt_Woerter'].values[0], 1))
        t_turns_length_mean_sd.set(round(stats.get().loc[stats.get()['Sprecher'] == input.name_teacher(), 'Median_Woerter'].values[0], 1))
        p_turns.set(stats.get().loc[stats.get()['Sprecher'] == "Schüler:innen", 'Anzahl_Beitraege'].values[0])
        p_turns_length.set(round(stats.get().loc[stats.get()['Sprecher'] == "Schüler:innen", 'Durchschnitt_Woerter'].values[0], 1))
        p_turns_length_mean_sd.set(stats.get().loc[stats.get()['Sprecher'] == "Schüler:innen", 'Median_Woerter'].values[0])


    # Anzeige der Gruppen-ID
    @render.ui
    def loc_group_id_display():
        return ui.value_box(
                        ui.p(t("analysis", "group_id")),
                        ui.output_text("nameGroup"),
                        showcase=icon_svg("id-card"),
                    ),
    

    @render.text
    def nameGroup():
        return input.name_group()


    # Anzeige der Klassengröße
    @render.ui
    def loc_class_size():
        return ui.value_box(
                        ui.p(t("results", "class_size")),
                        ui.output_text("numPupils"),
                        showcase=icon_svg("user-group"),
                    ),


    @render.text
    def numPupils():
        return input.num_pupils()
    

    # Anzeige der Anzahl beteiligter Schüler:innen
    @render.ui
    def loc_num_participants():
        return ui.value_box(
                        ui.p(t("results", "num_participants")),
                        ui.output_text("numParticipants"),
                        showcase=icon_svg("user-check"),
                    ),


    @render.text
    def numParticipants():
        req(num_participants.get() != None)
        return num_participants.get()
    

    # Anzeige der Beteiligungsquote
    @render.ui
    def loc_participation_rate():
        return ui.value_box(
                        ui.p(t("results", "participation_rate")),
                        ui.output_text("participationRate"),
                        showcase=icon_svg("square-poll-vertical"),
                    ),
    
    
    @render.text
    @reactive.calc
    def participationRate():
        req(num_participants.get() != None) 
        participation_rate.set(num_participants.get() / input.num_pupils() * 100)
        return f"{round(participation_rate.get(), 2)} %" 
    

    # Verteilung der Gesprächsbeiträge
    @render.ui
    def loc_distribution_of_turns():
        return ui.p(t("results", "distribution_of_turns"))
    

     # Create a bar plot for quantitative statistics
    @reactive.calc
    def make_sim_stats_plot():
        req(transcript_data.get() != None)

        distribution = stats.get().plot(kind='bar', x='Sprecher', y='Gesamt_Woerter', alpha=1, rot=0)
        plt.gca().set_xlabel(t("results", "words_total"))
        plt.gca().set_ylabel(t("results", "quantity"))
        distribution.set_axisbelow(True)
        distribution.grid(color='gray', axis = 'y')
        distribution.get_legend().remove()
        total = stats.get()['Gesamt_Woerter'].sum()
        distribution.set_xticklabels([t("stats", "teacher"), t("stats", "students")])
        for container in distribution.containers:
            perc_labels = [f"{(bar.get_height() / total * 100):.1f}%" for bar in container]

            distribution.bar_label(container, label_type='center')
            distribution.bar_label(container, labels=perc_labels, label_type='edge') 
            
        sim_plot.set(distribution)
        return distribution


    # Plot für Gesprächsverteilung
    @render.plot(alt="placeholder")
    def sim_stats_plot():
        if analysis_state.get() == False:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        else:
            return make_sim_stats_plot()  

    # Gesprächsstatistiken
    @render.ui
    def loc_interaction_turns():
        return ui.p(t("results", "interaction_turns"))
    
    # Lehrperson
    @render.ui
    def loc_teacher():
        return ui.p(t("results", "teacher"))


    @render.ui
    def loc_teacher_turns():
        return ui.markdown(f"**{t("results", "turn_count")}**")
    

    # Gesprächsbeiträge Lehrperson
    @render.text
    def teacher_turns():
        req(analysis_state.get(), transcript_data.get() != None)
        return stats.get().loc[stats.get()['Sprecher'] == input.name_teacher(), 'Anzahl_Beitraege'].values[0]
    

    # Display the number of teacher turns
    @render.text
    def teacher_impulses():
        req(teacher_impulses_count.get() != None)
        return teacher_impulses_count.get()
    

    @render.ui
    def loc_teacher_turns_length():
        return ui.markdown(f"**{t("results", "turn_length")}**")


    # Länge der Gesprächsbeiträge Lehrperson
    @render.text
    def teacher_turns_length():
        req(analysis_state.get(), transcript_data.get() != None)
        return f"{round(stats.get().loc[stats.get()['Sprecher'] == input.name_teacher(), 'Durchschnitt_Woerter'].values[0], 1)} ({round(stats.get().loc[stats.get()['Sprecher'] == input.name_teacher(), 'Median_Woerter'].values[0], 1)})"
    

    # Schüler:innen
    @render.ui
    def loc_pupils():
        return ui.p(t("results", "students"))
    

    @render.ui
    def loc_pupils_turns():
        return ui.markdown(f"**{t("results", "turn_count")}**")


    # Gesprächsbeiträge Schüler:innen
    @render.text
    def pupils_turns():
        req(analysis_state.get(), transcript_data.get() != None)
        return stats.get().loc[stats.get()['Sprecher'] == "Schüler:innen", 'Anzahl_Beitraege'].values[0]
    

    @render.ui
    def loc_pupils_turns_length():
        return ui.markdown(f"**{t("results", "turn_length")}**")
    

    # Länge der Gesprächsbeiträge Schüler:innen
    @render.text
    def pupils_turns_length():
        req(analysis_state.get(), transcript_data.get() != None)
        return f"{round(stats.get().loc[stats.get()['Sprecher'] == "Schüler:innen", 'Durchschnitt_Woerter'].values[0], 1)} ({stats.get().loc[stats.get()['Sprecher'] == "Schüler:innen", 'Median_Woerter'].values[0]
        })"
    

    # Anzeige der Qualitativen Analyse
    @render.ui
    def loc_qualitative_analysis():
        return ui.h3(t("results", "section_qualitative_analysis"))


    # Quick Stats
    @render.ui
    def loc_impulses_count():
        return ui.p(t("results", "impulses_count"))


    @render.ui
    def loc_coded_impulses():
        return ui.p(t("results", "coded_impulses"))
    

    # Display the number of impulses coded
    @render.text
    def teacher_impulses_coded():
        req(analysis_llm_state.get(), analysis_state.get())
        # Count number of rows in the dataframe
        num_impulses = qual_stats_df.get().shape[0] if qual_stats_df.get() is not None else "0"
        return num_impulses
    

    @render.ui
    def loc_most_frequent_codes():
        return ui.p(t("results", "most_frequent_codes"))
    
    # Display the most used code
    @render.text
    def code_most_used():
        req(analysis_llm_state.get(), analysis_state.get())
        # Find the most used code
        try:
            most_used_codes = qual_stats_df.get()[t("report", "shortcode")].mode().to_list() if not qual_stats_df.get().empty else t("system_prompts", "no_code")
            return ', '.join(most_used_codes)
        except:
            pass


    @render.ui
    def loc_teacher_talking_rate():
        return ui.p(t("results", "teacher_talking_rate"))
    

    # Display the share of words spoken by the teacher
    @render.text
    def teacher_share():
        req(stats.get().empty == False)
        # Calculate the share of words spoken by the teacher
        total_words = stats.get()['Gesamt_Woerter'].sum()
        teacher_words = stats.get().loc[stats.get()['Sprecher'] == input.name_teacher(), 'Gesamt_Woerter'].values[0]
        share = (teacher_words / total_words * 100) if total_words > 0 else 0
        return f"{round(share, 2)} %"


    # Qualitative Statistics Plot for Coded Impulses
    @render.ui
    def loc_impulses_distribution():
        ui.p(t("results", "impulses_distribution"))


    # Create a bar plot for qualitative statistics
    @reactive.calc
    def make_qualitative_stats_plot():
        req(llm_analysis_data.get())
        analysis_plot = llm_analysis_data.get()[-1].groupby(t("report", "shortcode")).agg(
            Anzahl=(t("report", "shortcode"), 'count'),
            ).reset_index().plot(kind='bar', x=t("report", "shortcode"), y='Anzahl', alpha=1, rot=0)
        plt.gca().set_xlabel(t("report", "shortcode"))
        plt.xticks(rotation=45, ha='right')
        plt.gca().set_ylabel(t("report", "quantity"))
        analysis_plot.set_axisbelow(True)
        analysis_plot.grid(color='gray', axis = 'y')
        analysis_plot.get_legend().remove()
        for container in analysis_plot.containers:
            analysis_plot.bar_label(container, label_type='edge')
        qual_plot.set(analysis_plot)
        return analysis_plot
    

    # Plot für qualitative Statistik
    @render.plot(alt="Noch keine Daten")
    def qualitative_stats_plot():
        if not llm_analysis_data.get():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        else:
            return make_qualitative_stats_plot() 
    

    # DataFrame of Coded Impulses
    @render.ui
    def loc_impulses_coding():
        ui.p(t("results", "impulses_coding"))


    # Create a DataFrame for qualitative statistics
    @reactive.calc
    def make_qualitative_stats_df():
        req(llm_analysis_data.get())
        analysis_df = llm_analysis_data.get()[-1]
        analysis_df['#'] = analysis_df.reset_index().index+1
        analysis_df = analysis_df[['#', "Impuls", "Shortcode"]]
        analysis_df.columns = ['#', t("report", "teacher_statement"), t("report", "shortcode")]
        qual_stats_df.set(analysis_df)
        return analysis_df        
    

# DataFrame für qualitative Statistik generieren
    @render.table()
    def qualitative_stats_df():
        return make_qualitative_stats_df()


    # DataFrame für qualitative Statistik
    @render.ui
    def quali_stats_df():
        if not llm_analysis_data.get():
            return ui.output_plot("placeholder")
        else:
            return ui.output_table("qualitative_stats_df")


    # Placeholder Plot, wenn noch keine Daten vorhanden sind
    @render.plot(alt="Noch keine Daten")
    def placeholder():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
        ax.axis('off')
        placeholder_plot.set(fig)
        return fig

    
    @render.plot(alt="Noch keine Daten")
    def placeholder2():
        return placeholder_plot.get()


    # Code-Legende aus Codebuch extrahieren
    @reactive.effect
    def extract_code_legend():
        req(codebook_data.get() != None)
        df = pd.DataFrame(codebook_data.get())
        legend = []
        for code in df[df.columns[0]].unique():
            legend.append(f"{code}")
        code_legend_storage.set("; ".join(legend))
    

    # Code-Legende anzeigen
    @render.ui
    def code_legend():
        return ui.markdown(f"**{t("results", "caption")}:** {code_legend_storage.get()}")


    ### Optionen --------------------------------------------------------

    @render.text
    def loc_title_options():
        return (t("options", "tab_title"))
    

    # Api Konfiguration
    @render.ui
    def loc_api_configuration():
        return ui.p(t("options", "api_configuration"))
    

    @render.text
    def loc_api_select_title():
        return t("options", "api_select_title")


    @render.ui
    def loc_api_select():
        return ui.input_switch("api_select", t("options", "api_select"), True)
    
    @reactive.effect
    def update_api_selection():
        config.set_current_api("openai" if input.api_select() else "groq")
        current_api.set("openai" if input.api_select() else "groq")

    # Anzeige, ob ein API-Key vorhanden ist
    @render.text
    def loc_api_key_exists():
        if not input.api_select():
            a = api_key_groq.get() # for reactivity/invalidation
            return t("options", "api_groq_found") if api_key_groq.get() else t("options", "api_groq_not_found")
        else:
            a = api_key_openai.get() # for reactivity/invalidation
            return t("options", "api_openai_found") if api_key_openai.get() else t("options", "api_openai_not_found")

    # API-Auswahl
    @reactive.calc
    def select_api_choices():
        deleted_model = model_deleted.get() # for reactivity/invalidation
        api_current = current_api.get() # for reactivity/invalidation
        if config.get_current_api() == "openai":
            return config.get_models(provider="openai")
        if config.get_current_api() == "groq":
            return config.get_models(provider="groq")
        
    # Warnung bei fehlendem API-Key   
    @reactive.effect
    @reactive.event(input.button_analysis)
    def _():
        req(input.llm_switch(), input.button_analysis(), transcript_data.get() != None, codebook_data.get() != None)
        if api_key_openai.get() == None and input.api_select() or api_key_groq.get() == None and not input.api_select():
            m = ui.modal(  
                    ui.p(t("options", "no_api_key_warning")),  
                    title=t("analysis", "modal_title_error"),  
                    easy_close=True,
                    footer=ui.modal_button(t("analysis", "modal_button_close")), 
                )
            ui.modal_show(m)
            ui.update_navs("main_tabs", selected='<div id="loc_title_options" class="shiny-text-output"></div>')  

    # Button zum Ändern des API-Keys
    @render.ui
    def loc_button_change_api_key():
        return ui.input_action_button("button_change_api_key", t("options", "button_change"), icon=icon_svg("wrench")),
    

    @reactive.effect
    @reactive.event(input.button_change_api_key)
    def change_api_key():
        m = ui.modal(
            ui.input_password("api_key", label=None, placeholder=t("options", "add_api_key_placeholder")),
            title=t("options", "add_api_key_title"),
            easy_close=True,
            footer=(ui.input_action_button("button_save_api_key", t("options", "add_api_key_save"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"), class_="btn-danger")),
        )
        ui.modal_show(m)

    # Speichern des API-Keys
    @reactive.effect
    @reactive.event(input.button_save_api_key)
    def save_api_key():
        req(input.api_key())
        if input.api_select():
            keyring.set_password("talktrace", "api_key_openai", input.api_key())
            api_key_openai.set(input.api_key())
        else:
            keyring.set_password("talktrace", "api_key_groq", input.api_key())
            api_key_groq.set(input.api_key())
        ui.modal_remove()   


    @render.ui
    def loc_button_delete_api_key():
        return ui.input_action_button("button_delete_api_key", t("options", "button_delete"), icon = icon_svg("trash-can"), class_="btn-danger"),


   # Button zum Löschen des API-Keys
    @reactive.effect
    @reactive.event(input.button_delete_api_key)
    def delete_api_key():
        m = ui.modal(
            ui.p(t("options", "delete_api_key_warning")),
            title=t("options", "delete_api_key_title"),
            easy_close=True,
            footer=(ui.input_action_button("button_confirm_delete_api_key", t("options", "delete_api_key_confirm"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger")),
        )
        ui.modal_show(m)


    # Löschens des API-Keys bestätigen
    @reactive.effect
    @reactive.event(input.button_confirm_delete_api_key)
    def confirm_delete_api_key():
        try:
            keyring.delete_password("talktrace", "api_key_openai") if input.api_select() else keyring.delete_password("talktrace", "api_key_groq")
        except keyring.errors.PasswordDeleteError:
            pass
        
        api_key_openai.set(None) if input.api_select() else api_key_groq.set(None)
        ui.modal_remove()


    # Modelle für LLM-Auswahl
    @render.ui
    def loc_llm_models():
        return ui.p(t("options", "llm_models"))
    

    # Verfügbare Modelle aus Config laden und auflisten
    @render.ui
    def loc_load_models():
        return ui.input_select("model_list", t("options", "available_models"), choices=models_available(), multiple=True)
    
    @reactive.calc
    def models_available():
        deleted_models = model_deleted.get() # for reactivity/invalidation
        return config.get_models()

    # Button zum Hinzufügen eines Modells
    @render.ui
    def loc_button_add_model():
        return ui.input_action_button("button_add_model", t("options", "add_model"), icon = icon_svg("plus"), class_="btn-success"),

    # Modal zum Hinzufügen eines Modells
    @reactive.effect
    @reactive.event(input.button_add_model)
    def add_model():
        m = ui.modal(
            ui.input_text("model_id", t("options", "model_id"), placeholder=t("options", "add_model_placeholder")),
            ui.input_select("model_provider", t("options", "model_provider"), choices=["openai", "groq"], selected="openai"),
            ui.input_text("intput_cost", t("options", "input_cost"), placeholder=t("options", "cost_placeholder")),
            ui.input_text("output_cost", t("options", "output_cost"), placeholder=t("options", "cost_placeholder")),
            title=t("options", "add_model_title"),
            easy_close=True,
            footer=(ui.input_action_button("model_add_confirm", t("options", "modal_button_add"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger")),
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.model_add_confirm)
    def confirm_add_model():
        req(input.model_id(), input.model_provider())
        config.add_model(input.model_provider(), input.model_id(), float(input.intput_cost()), float(input.output_cost()))
        # Update available models in the model options
        available_models = config.get_models()
        model_deleted.set(model_deleted.get() + 1) # for reactivity/invalidation
        ui.update_select("model_list", choices=available_models)
        ui.update_select("model_select", choices=select_api_choices())
        ui.modal_remove()
    
    # Button zum Entfernen eines Modells
    @render.ui
    def loc_button_remove_model():
        return ui.input_action_button("button_remove_model", t("options", "remove_model"), icon = icon_svg("trash-can"), class_="btn-danger"),

    # Modal zum Entfernen von Modellen
    @reactive.effect
    @reactive.event(input.button_remove_model)
    def _():
        m = ui.modal(
        ui.p(t("options", "modal_remove_model_warning")),
        title=t("options", "modal_remove_title"),
        easy_close=True,
        footer=(ui.input_action_button("model_delete_confirm", t("options", "modal_remove_confirm"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger"))
        )
        ui.modal_show(m)

    # Entfernen des Modells bestätigen
    @reactive.effect
    @reactive.event(input.model_delete_confirm)
    def _():
        config.remove_model(list(input.model_list()))
        # Update available models in the model options
        model_deleted.set(model_deleted.get() + 1) # for reactivity/invalidation
        available_models = config.get_models()
        ui.update_select("model_list", choices=available_models)
        # If current selected model was removed, update model selection
        if input.model_select() not in available_models:
            ui.update_select("model_select", choices=select_api_choices())
        ui.modal_remove()


    # Modell-Auswahl auf Default zurücksetzen
    @render.ui
    def loc_button_reset_model_selection():
        return ui.input_action_button("button_reset_model_selection", t("options", "button_reset"), icon = icon_svg("arrow-rotate-left"), class_="btn-danger"),


    # Modal zum Zurücksetzen der Modellauswahl
    @reactive.effect
    @reactive.event(input.button_reset_model_selection)
    def reset_model_selection():
        m = ui.modal(
            ui.p(t("options", "reset_model_selection_confirm")),
        title=t("options", "reset_model_selection_title"),
        easy_close=True,
        footer=(ui.input_action_button("button_reset_model_selection_confirm", t("options", "modal_model_reset_confirm"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger")),
        )
        ui.modal_show(m)

    # Zurücksetzen der Modellauswahl bestätigen
    @reactive.effect
    @reactive.event(input.button_reset_model_selection_confirm)
    def confirm_reset_model_selection():
        config.reset_models()
      # ui.update_select("model_select", choices=select_api_choices())
        model_deleted.set(model_deleted.get() + 1) # for reactivity/invalidation
        ui.modal_remove()

    # Benutzerdefinierte Prompts
    @render.ui
    def loc_custom_prompts():
        return ui.p(t("options", "custom_prompts"))
    
    # System Prompt anzeigen
    @render.text()
    def system_prompt_output():
        return system_prompt.get()
    
    # Button zum Ändern des System Prompts
    @render.ui
    def loc_button_change_system_prompt():
        return ui.input_action_button("button_change_system_prompt", t("options", "button_change"), icon = icon_svg("pen")),
    
    
    @reactive.effect
    @reactive.event(input.button_change_system_prompt)
    def change_system_prompt():
        m = ui.modal(
            ui.input_text_area("system_prompt", t("options", "change_system_prompt"), system_prompt.get(), rows=10),
            title=t("options", "change_system_prompt"),
            easy_close=True,
            footer=(ui.input_action_button("button_save_system_prompt", t("options", "add_api_key_save"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger")),
        )
        ui.modal_show(m)

    # Speichern des System Prompts
    @reactive.effect
    @reactive.event(input.button_save_system_prompt)
    def save_system_prompt():
        req(input.system_prompt())
        config.set_prompt('system', input.system_prompt())
        system_prompt.set(input.system_prompt())
        ui.modal_remove()

    
    @render.ui
    def loc_button_reset_system_prompt():
        return ui.input_action_button("button_reset_system_prompt", t("options", "button_reset"), icon = icon_svg("arrow-rotate-left"), class_="btn-danger"),


    # Button zum Zurücksetzen des System Prompts
    @reactive.effect
    @reactive.event(input.button_reset_system_prompt)
    def reset_system_prompt():
        m = ui.modal(
            ui.p(t("options", "reset_system_prompt_confirm")),
        title=t("options", "reset_system_prompt_title"),
        easy_close=True,
        footer=(ui.input_action_button("button_reset_system_prompt_confirm", t("analysis", "modal_confirm_reset"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger")),
        )
        ui.modal_show(m)

    # Zurücksetzen des System Prompts Bestätigen
    @reactive.effect
    @reactive.event(input.button_reset_system_prompt_confirm)
    def confirm_reset_system_prompt():
        config.set_prompt('system', config.get_prompts()['system_default'])
        system_prompt.set(config.get_prompts()['system'])
        ui.modal_remove()


    # User Prompt anzeigen
    @render.text()
    def user_prompt_output():
        return user_prompt.get()
    

    @render.ui
    def loc_button_change_user_prompt():
        return ui.input_action_button("button_change_user_prompt", t("options", "button_change"), icon = icon_svg("pen")),


    # Button zum Ändern des User Prompts
    @reactive.effect
    @reactive.event(input.button_change_user_prompt)
    def change_user_prompt():
        m = ui.modal(
            ui.input_text_area("user_prompt", t("options", "change_user_prompt"), user_prompt.get(), rows=10),
            title=t("options", "change_user_prompt"),
            easy_close=True,
            footer=(ui.input_action_button("button_save_user_prompt", t("options", "add_api_key_save"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"), class_="btn-danger")),
        )
        ui.modal_show(m)


    # Speichern des User Prompts
    @reactive.effect
    @reactive.event(input.button_save_user_prompt)
    def save_user_prompt():
        req(input.user_prompt())
        config.set_prompt('user', input.user_prompt())
        user_prompt.set(input.user_prompt())
        ui.modal_remove()   


    # Button zum Zurücksetzen des User Prompts
    @render.ui
    def loc_button_reset_user_prompt():
        return ui.input_action_button("button_reset_user_prompt", t("options", "button_reset"), icon = icon_svg("arrow-rotate-left"), class_="btn-danger"),


    @reactive.effect
    @reactive.event(input.button_reset_user_prompt)
    def reset_user_prompt():
        m = ui.modal(
            ui.p(t("options", "reset_user_prompt_confirm")),
        title=t("options", "reset_user_prompt_title"),
        easy_close=True,
        footer=(ui.input_action_button("button_reset_user_prompt_confirm", t("analysis", "modal_confirm_reset"), class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"), class_="btn-danger")),
        )
        ui.modal_show(m)

    # Zurücksetzen des User Prompts Bestätigen
    @reactive.effect
    @reactive.event(input.button_reset_user_prompt_confirm)
    def confirm_reset_user_prompt():
        config.set_prompt('user', config.get_prompts()['user_default'])
        user_prompt.set(config.get_prompts()['user'])
        ui.modal_remove()

    # Weitere Optionen
    @render.ui
    def loc_additional_options():
        return ui.p(t("options", "additional_options"))
    

    @render.ui
    def loc_input_teacher_name_options():
        return ui.input_text("name_teacher_options", t("options", "teacher_name"), config.get_parameters()['teacher_name'])
    
    # Parameter in Config Speichern
    @reactive.effect
    @reactive.event(input.name_teacher_options)
    def _():
        config.set_parameter('teacher_name', input.name_teacher_options())


    @render.ui
    def loc_input_group_id_options():
        return ui.input_text("name_group_options", t("options", "group_id"), "B1")
    
    @reactive.effect
    @reactive.event(input.name_group_options)
    def _():
        config.set_parameter('group_id', input.name_group_options())


    @render.ui
    def loc_input_num_pupils_options():
        return ui.input_numeric("num_pupils_options", t("options", "num_students"), 25, min=1, max=100)


    @reactive.effect
    @reactive.event(input.num_pupils_options)
    def _():
        config.set_parameter('num_pupils', input.num_pupils_options())


    @render.ui
    def loc_button_reset_parameters():
        return ui.input_action_button("button_reset_parameters", t("options", "button_reset"), icon = icon_svg("arrow-rotate-left"), class_="btn-danger"),
    

    # Button zum Zurücksetzen der Gruppen-Parameter
    @reactive.effect
    @reactive.event(input.button_reset_parameters)
    def reset_user_prompt():
        m = ui.modal(
            ui.p(t("options", "reset_group_parameters_confirm")),
        title=t("options", "reset_group_parameters_title"),
        easy_close=True,
        footer=(ui.input_action_button("button_reset_parameters_confirm", t("analysis", "modal_confirm_reset"), class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"), class_="btn-danger")),
        )
        ui.modal_show(m)

    # Zurücksetzen der Gruppen-Parameter bestätigen
    @reactive.effect
    @reactive.event(input.button_reset_parameters_confirm)
    def confirm_reset_parameters():
        config.set_parameter('teacher_name', config.get_parameters()['teacher_name_default'])
        config.set_parameter('group_id', config.get_parameters()['group_id_default'])
        config.set_parameter('num_pupils', config.get_parameters()['num_pupils_default'])
        ui.update_text("name_teacher_options", value=config.get_parameters()['teacher_name'])
        ui.update_text("name_group_options", value=config.get_parameters()['group_id'])
        ui.update_numeric("num_pupils_options", value=config.get_parameters()['num_pupils'])
        ui.modal_remove()

    # About TalkTrace AI
    @render.ui
    def loc_app_info():
        return ui.p(t("options", "about"))
    
    @render.ui
    def loc_app_info_text():
        return ui.markdown(t("options", "about_text"))

# -----------------------------------------------------------------------------------------------------------   

# App als globales Objekt initiasieren, damit der server zugreifen kann
app = App(app_ui, server, debug=False)

# Get the directory containing the current file
current_dir = Path(__file__).parent

def main():
    run_app(app)
