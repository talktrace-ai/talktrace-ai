"""Options section: API keys, model registry, prompts, parameters."""
from ._common import *


def register(state):
    input = state.input
    output = state.output
    session = state.session
    config = state.config
    t = state.t
    transcript_data = state.transcript_data
    codebook_data = state.codebook_data
    api_key_groq = state.api_key_groq
    api_key_openai = state.api_key_openai
    api_key_anthropic = state.api_key_anthropic
    api_key_ollama = state.api_key_ollama
    ollama_status_refresh = state.ollama_status_refresh
    current_api = state.current_api
    model_deleted = state.model_deleted
    system_prompt = state.system_prompt
    user_prompt = state.user_prompt

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


    def _options_provider_choices():
        if state.local_only.get():
            return {"ollama": "Ollama"}
        return {"openai": "OpenAI", "groq": "Groq", "anthropic": "Anthropic", "ollama": "Ollama"}

    @render.ui
    def loc_api_select():
        return ui.input_select("api_select", t("options", "api_select_title"), choices=_options_provider_choices(), selected=config.get_current_api())

    @reactive.effect
    def update_api_selection():
        config.set_current_api(input.api_select())
        current_api.set(input.api_select())

    # Anzeige, ob ein API-Key vorhanden ist
    @render.text
    def loc_api_key_exists():
        selected = input.api_select()
        if selected == "openai":
            a = api_key_openai.get()
            return t("options", "api_openai_found") if api_key_openai.get() else t("options", "api_openai_not_found")
        elif selected == "groq":
            a = api_key_groq.get()
            return t("options", "api_groq_found") if api_key_groq.get() else t("options", "api_groq_not_found")
        elif selected == "anthropic":
            a = api_key_anthropic.get()
            return t("options", "api_anthropic_found") if api_key_anthropic.get() else t("options", "api_anthropic_not_found")
        elif selected == "ollama":
            ollama_status_refresh.get()  # reactivity
            reactive.invalidate_later(600)
            url = "http://localhost:11434/"
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    if resp.status == 200:
                        return t("options", "ollama_running").format(url=url)
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            return t("options", "ollama_not_running").format(url=url)

    # API-Auswahl
    @reactive.calc
    def select_api_choices():
        deleted_model = model_deleted.get() # for reactivity/invalidation
        api_current = current_api.get() # for reactivity/invalidation
        # Local-only filters out cloud models (`:cloud` suffix or explicit
        # `local: false`) so the user cannot accidentally route to a cloud
        # endpoint hosted by Ollama Inc. or a similar third party.
        return config.get_models(
            provider=config.get_current_api(),
            local_only=state.local_only.get(),
        )

    state.select_api_choices = select_api_choices

    # Warnung bei fehlendem API-Key
    @reactive.effect
    @reactive.event(input.button_analysis)
    def _():
        req(input.llm_switch(), input.button_analysis(), transcript_data.get() != None, codebook_data.get() != None)
        selected = input.api_select()
        missing_key = (
            (selected == "openai" and api_key_openai.get() == None) or
            (selected == "groq" and api_key_groq.get() == None) or
            (selected == "anthropic" and api_key_anthropic.get() == None)
        )
        if missing_key:
            m = ui.modal(
                    ui.p(t("options", "no_api_key_warning")),
                    title=t("analysis", "modal_title_error"),
                    easy_close=True,
                    footer=ui.modal_button(t("analysis", "modal_button_close")),
                )
            ui.modal_show(m)
            ui.update_navset("main_tabs", selected='<div id="loc_title_options" class="shiny-text-output"></div>')

    # Button zum Ändern des API-Keys
    @render.ui
    def loc_button_change_api_key():
        if input.api_select() == "ollama":
            return ui.input_action_button("button_change_api_key", t("options", "ollama_start_button"), icon=icon_svg("play")),
        return ui.input_action_button("button_change_api_key", t("options", "button_change"), icon=icon_svg("wrench")),


    @reactive.effect
    @reactive.event(input.button_change_api_key)
    async def change_api_key():
        if input.api_select() == "ollama":
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
                )
            except FileNotFoundError:
                ui.modal_show(ui.modal(
                    ui.p(t("options", "ollama_start_error")),
                    title=t("options", "error_title"),
                    easy_close=True,
                    footer=ui.modal_button(t("analysis", "modal_button_close")),
                ))
                return
            ollama_status_refresh.set(ollama_status_refresh.get() + 1)

            async def _delayed_refresh():
                await asyncio.sleep(5)
                async with reactive.lock():
                    ollama_status_refresh.set(ollama_status_refresh.get() + 1)
                    await reactive.flush()
            asyncio.create_task(_delayed_refresh())
            return
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
        selected = input.api_select()
        key_for = {
            "openai": "api_key_openai",
            "groq": "api_key_groq",
            "anthropic": "api_key_anthropic",
            "ollama": "api_key_ollama",
        }
        target_for = {
            "openai": api_key_openai,
            "groq": api_key_groq,
            "anthropic": api_key_anthropic,
            "ollama": api_key_ollama,
        }
        if selected in key_for:
            persisted = safe_set_password("talktrace", key_for[selected], input.api_key())
            target_for[selected].set(input.api_key())
            if not persisted:
                ui.notification_show(
                    t("options", "keyring_unavailable"),
                    type="warning",
                    duration=8,
                )
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
        selected = input.api_select()
        key_for = {
            "openai": "api_key_openai",
            "groq": "api_key_groq",
            "anthropic": "api_key_anthropic",
            "ollama": "api_key_ollama",
        }
        target_for = {
            "openai": api_key_openai,
            "groq": api_key_groq,
            "anthropic": api_key_anthropic,
            "ollama": api_key_ollama,
        }
        if selected in key_for:
            safe_delete_password("talktrace", key_for[selected])
            target_for[selected].set(None)
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
        return config.get_models(local_only=state.local_only.get())

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
            ui.input_select("model_provider", t("options", "model_provider"), choices=["openai", "groq", "anthropic", "ollama"], selected="openai"),
            ui.input_text("intput_cost", t("options", "input_cost"), placeholder=t("options", "cost_placeholder")),
            ui.input_text("output_cost", t("options", "output_cost"), placeholder=t("options", "cost_placeholder")),
            ui.input_checkbox("model_is_local", t("options", "model_is_local"), value=False),
            ui.tags.p(t("options", "model_is_local_hint"), class_="text-muted small"),
            title=t("options", "add_model_title"),
            easy_close=True,
            footer=(ui.input_action_button("model_add_confirm", t("options", "modal_button_add"),  class_="btn-success"), ui.modal_button(t("analysis", "modal_button_cancel"),  class_="btn-danger")),
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.model_add_confirm)
    def confirm_add_model():
        req(input.model_id(), input.model_provider())

        def _parse_cost(raw: str) -> float:
            try:
                return float(raw.strip().replace(",", "."))
            except (ValueError, AttributeError):
                return 0.0

        input_cost = _parse_cost(input.intput_cost())
        output_cost = _parse_cost(input.output_cost())

        try:
            local_flag = bool(input.model_is_local())
        except Exception:
            local_flag = None  # fall back to provider/suffix heuristic
        config.add_model(input.model_provider(), input.model_id(),
                         input_cost, output_cost, local=local_flag)
        available_models = config.get_models(local_only=state.local_only.get())
        model_deleted.set(model_deleted.get() + 1)
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

    # System Prompt anzeigen (effektive Version inkl. Sprecher-Filter)
    @render.text()
    def system_prompt_output():
        return state.effective_system_prompt()

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


    # User Prompt anzeigen (aktuell ohne Sprecher-Filter, aber via effective_*-Getter
    # konsistent gehalten, falls später zusätzlich angepasst werden soll)
    @render.text()
    def user_prompt_output():
        return state.effective_user_prompt()


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
        return ui.input_text("name_group_options", t("options", "group_id"), config.get_parameters()['group_id'])

    @reactive.effect
    @reactive.event(input.name_group_options)
    def _():
        config.set_parameter('group_id', input.name_group_options())


    @render.ui
    def loc_input_num_pupils_options():
        return ui.input_numeric("num_pupils_options", t("options", "num_students"), config.get_parameters()['num_pupils'], min=1, max=100)


    @reactive.effect
    @reactive.event(input.num_pupils_options)
    def _():
        config.set_parameter('num_pupils', input.num_pupils_options())


    @render.ui
    def loc_button_reset_parameters():
        # Unsichtbares Spacer-Label mit derselben Struktur wie die Input-Labels
        # in den Nachbarspalten, damit der Button auf gleicher Höhe wie die
        # Eingabefelder sitzt. Die Vertikal-Paddings am Button selbst werden
        # an die Form-Control-Höhe angeglichen — sonst ist der Button höher
        # als die Inputs und ragt oben hinaus.
        return ui.div(
            ui.tags.label(
                " ",
                class_="control-label",
                style="display: block; visibility: hidden;",
            ),
            ui.input_action_button(
                "button_reset_parameters",
                t("options", "button_reset"),
                icon=icon_svg("arrow-rotate-left"),
                class_="btn-danger",
                style="padding-top: 0.25rem; padding-bottom: 0.25rem; margin-top: 2px;",
            ),
        ),


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

    # Erweitert: Streaming-Toggle (liest/schreibt ADVANCED.streaming in der Config;
    # die Sidebar liest denselben Schlüssel beim Klick auf "Analysieren", daher
    # genügt ein Config-Round-Trip — kein eigener Reactive-Wert nötig.
    @render.ui
    def loc_advanced_options():
        return ui.p(t("options", "advanced_options"))

    @render.ui
    def loc_streaming_switch():
        return ui.div(
            ui.input_switch(
                "streaming_switch",
                t("options", "streaming_switch"),
                config.get_advanced().get("streaming", False),
            ),
            ui.tags.p(t("options", "streaming_switch_help"), class_="text-muted small"),
        )

    @reactive.effect
    @reactive.event(input.streaming_switch)
    def _persist_streaming_switch():
        config.set_advanced("streaming", bool(input.streaming_switch()))

    @render.ui
    def loc_local_only_switch():
        return ui.div(
            ui.input_switch(
                "local_only_switch",
                t("options", "local_only_switch"),
                config.get_advanced().get("local_only", False),
            ),
            ui.tags.p(t("options", "local_only_switch_help"), class_="text-muted small"),
        )

    # ----- Cumulative cost tracker ----------------------------------------
    @render.ui
    def loc_cost_tracker_header():
        return ui.p(t("options", "cost_tracker_header"))

    def _format_eur(amount, lang):
        s = f"{amount:.2f}"
        return s.replace(".", ",") if lang == "de" else s

    @render.ui
    def cost_tracker_table():
        # Reactive on the version bump from analysis runs + the reset button.
        state.cost_tracker_version.get()
        summary = get_cost_summary()
        try:
            lang = state.current_lang.get()
        except Exception:
            lang = "en"
        if not summary["n_runs"]:
            return ui.p(t("options", "cost_tracker_empty"), class_="text-muted")

        total = _format_eur(summary["total_cost"], lang)
        rows = [
            ui.tags.tr(
                ui.tags.th(t("options", "cost_tracker_total"), colspan=2),
                ui.tags.td(f"{total} €", style="text-align:right;font-weight:600;"),
                ui.tags.td(f"{summary['n_runs']}", style="text-align:right;"),
            ),
        ]
        for provider, info in sorted(summary["by_provider"].items(), key=lambda kv: -kv[1]["cost"]):
            rows.append(ui.tags.tr(
                ui.tags.th(provider, scope="row"),
                ui.tags.td("—", class_="text-muted"),
                ui.tags.td(f"{_format_eur(info['cost'], lang)} €", style="text-align:right;"),
                ui.tags.td(f"{info['runs']}", style="text-align:right;"),
            ))
        for model_name, info in sorted(summary["by_model"].items(), key=lambda kv: -kv[1]["cost"]):
            rows.append(ui.tags.tr(
                ui.tags.td(info.get("provider", "—"), class_="text-muted small"),
                ui.tags.td(model_name),
                ui.tags.td(f"{_format_eur(info['cost'], lang)} €", style="text-align:right;"),
                ui.tags.td(f"{info['runs']}", style="text-align:right;"),
            ))
        header = ui.tags.thead(ui.tags.tr(
            ui.tags.th(t("options", "cost_tracker_col_provider")),
            ui.tags.th(t("options", "cost_tracker_col_model")),
            ui.tags.th(t("options", "cost_tracker_col_cost"), style="text-align:right;"),
            ui.tags.th(t("options", "cost_tracker_col_runs"), style="text-align:right;"),
        ))
        return ui.tags.table(header, ui.tags.tbody(*rows),
                             class_="table table-sm table-striped")

    @render.ui
    def loc_cost_tracker_reset_button():
        state.cost_tracker_version.get()
        if not get_cost_summary()["n_runs"]:
            return None
        return ui.input_action_button(
            "button_cost_tracker_reset",
            t("options", "cost_tracker_reset"),
            icon=icon_svg("trash-can"),
            class_="btn-sm btn-outline-danger",
        )

    @reactive.effect
    @reactive.event(input.button_cost_tracker_reset)
    def _confirm_cost_reset():
        ui.modal_show(ui.modal(
            ui.p(t("options", "cost_tracker_reset_confirm")),
            title=t("options", "cost_tracker_reset_title"),
            easy_close=True,
            footer=(
                ui.input_action_button("button_cost_tracker_reset_confirm",
                                       t("analysis", "modal_confirm_reset"),
                                       class_="btn-success"),
                ui.modal_button(t("analysis", "modal_button_cancel"),
                                class_="btn-danger"),
            ),
        ))

    @reactive.effect
    @reactive.event(input.button_cost_tracker_reset_confirm)
    def _do_cost_reset():
        reset_cost_log()
        state.cost_tracker_version.set(state.cost_tracker_version.get() + 1)
        ui.modal_remove()

    # ----- /Cumulative cost tracker --------------------------------------

    # ----- Gold-standard self-test ---------------------------------------
    @render.ui
    def loc_self_test_header():
        return ui.p(t("options", "self_test_header"))

    @render.ui
    def loc_self_test_intro():
        return ui.tags.p(t("options", "self_test_intro"), class_="text-muted small")

    @render.ui
    def loc_self_test_button():
        return ui.input_action_button(
            "button_self_test_run",
            t("options", "self_test_run"),
            icon=icon_svg("flask"),
            class_="btn-sm btn-primary",
        )

    @reactive.effect
    @reactive.event(input.button_self_test_run)
    def _do_self_test():
        try:
            lang = state.current_lang.get()
        except Exception:
            lang = "en"
        try:
            res = run_self_test(lang=lang)
        except Exception as exc:
            res = {
                "checks": [{
                    "label": "self_test failed to start",
                    "status": "fail",
                    "expected": "—",
                    "actual": str(exc),
                    "detail": "",
                }],
                "n_pass": 0,
                "n_total": 1,
                "all_pass": False,
            }
        state.self_test_result.set(res)

    @render.ui
    def self_test_result():
        res = state.self_test_result.get()
        if res is None:
            return None
        n_pass = res["n_pass"]
        n_total = res["n_total"]
        if res["all_pass"]:
            banner = ui.div(
                icon_svg("circle-check"),
                f" {t('options', 'self_test_all_pass').format(n=n_total)}",
                class_="alert alert-success",
                style="margin-top:0.75rem;",
            )
        else:
            banner = ui.div(
                icon_svg("triangle-exclamation"),
                f" {t('options', 'self_test_some_fail').format(passed=n_pass, total=n_total)}",
                class_="alert alert-danger",
                style="margin-top:0.75rem;",
            )
        rows = []
        for c in res["checks"]:
            ok = c["status"] == "pass"
            icon = icon_svg("check") if ok else icon_svg("xmark")
            rows.append(ui.tags.tr(
                ui.tags.td(icon, style=("color:#2a7;" if ok else "color:#c33;")),
                ui.tags.td(c["label"]),
                ui.tags.td(str(c["expected"]), class_="text-muted small"),
                ui.tags.td(str(c["actual"]), class_="small"),
            ))
        header = ui.tags.thead(ui.tags.tr(
            ui.tags.th(""),
            ui.tags.th(t("options", "self_test_col_check")),
            ui.tags.th(t("options", "self_test_col_expected")),
            ui.tags.th(t("options", "self_test_col_actual")),
        ))
        return ui.div(
            banner,
            ui.tags.table(header, ui.tags.tbody(*rows),
                          class_="table table-sm table-striped",
                          style="margin-top:0.5rem;"),
        )
    # ----- /Gold-standard self-test --------------------------------------

    @reactive.effect
    @reactive.event(input.local_only_switch)
    def _persist_local_only_switch():
        new_val = bool(input.local_only_switch())
        config.set_advanced("local_only", new_val)
        state.local_only.set(new_val)
        # When the user enables local-only and a cloud provider is selected,
        # snap to ollama so subsequent analysis cannot route to a cloud API.
        if new_val and config.get_current_api() != "ollama":
            config.set_current_api("ollama")
            current_api.set("ollama")
            ui.update_select("api_select", choices={"ollama": "Ollama"}, selected="ollama")
            ui.update_select("provider_select", choices={"ollama": "Ollama"}, selected="ollama")
        # Even on the same provider, the model dropdown may still point at a
        # cloud model (e.g. Ollama Cloud). Snap the selected model to the
        # first remaining local one so the next "Analyze" click cannot leak.
        if new_val:
            local_models = config.get_models(provider="ollama", local_only=True)
            current_model = config.get_current_model()
            if local_models and current_model not in local_models:
                config.set_current_model(local_models[0])
                state.model.set(local_models[0])
                ui.update_select("model_select", choices=local_models, selected=local_models[0])


