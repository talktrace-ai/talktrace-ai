"""Provider/model dropdowns + sync effects + Ollama hint."""
from .._common import *


def register(state):
    input = state.input
    config = state.config
    t = state.t
    current_api = state.current_api
    model = state.model

    @render.ui
    def loc_dynamic_model_select():
        return ui.div(
            ui.input_select("provider_select", t("sidebar", "provider_select"), choices={"openai": "OpenAI", "groq": "Groq", "anthropic": "Anthropic", "ollama": "Ollama"}, selected=config.get_current_api()),
            ui.input_select("model_select", t("sidebar", "model_select"), choices=state.select_api_choices(), selected=config.get_current_model()),
            **{"data-tt-help": t("onboarding", "tooltip_model_select")},
        )

    # Hinweis nur bei aktivem Ollama-Provider — als Tooltip auf einem
    # kleinen Info-Icon, damit die Sidebar nicht durch eine zusätzliche
    # Textzeile aufgebläht wird. Hover zeigt den Volltext.
    @render.ui
    def loc_ollama_hint():
        try:
            if input.provider_select() != "ollama":
                return None
        except Exception:
            return None
        return ui.tooltip(
            ui.tags.span(
                icon_svg("circle-info"),
                " ", t("sidebar", "ollama_cloud_hint_label"),
                class_="text-muted small",
                style="cursor: help;",
            ),
            t("sidebar", "ollama_cloud_hint"),
            placement="right",
        )

    @reactive.effect()
    def update_current_provider():
        selected_provider = input.provider_select()
        if not selected_provider:
            return
        if selected_provider == config.get_current_api():
            return
        config.set_current_api(selected_provider)
        current_api.set(selected_provider)
        # passendes erstes Modell des neuen Anbieters auswählen
        available_models = state.select_api_choices()
        if available_models:
            first_model = next(iter(available_models))
            model.set(first_model)
            config.set_current_model(first_model)
            ui.update_select("model_select", choices=available_models, selected=first_model)

    @reactive.effect()
    def update_current_model():
        model.set(input.model_select())
        config.set_current_model(input.model_select())
