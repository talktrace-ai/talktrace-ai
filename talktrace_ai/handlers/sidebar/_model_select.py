"""Provider/model dropdowns + sync effects + provider hint."""
from .._common import *


def register(state):
    input = state.input
    config = state.config
    t = state.t
    current_api = state.current_api
    model = state.model

    def _provider_choices():
        if state.local_only.get():
            return {"ollama": "Ollama"}
        return {"openai": "OpenAI", "groq": "Groq", "anthropic": "Anthropic", "ollama": "Ollama"}

    @render.ui
    def loc_dynamic_model_select():
        return ui.div(
            ui.input_select("provider_select", t("sidebar", "provider_select"), choices=_provider_choices(), selected=config.get_current_api()),
            ui.input_select("model_select", t("sidebar", "model_select"), choices=state.select_api_choices(), selected=config.get_current_model()),
            **{"data-tt-help": t("onboarding", "tooltip_model_select")},
        )

    # Hinweis je nach Provider — als Tooltip auf einem kleinen Info-Icon,
    # damit die Sidebar nicht durch eine zusätzliche Textzeile aufgebläht
    # wird. Hover zeigt den Volltext.
    _PROVIDER_HINTS = {
        "ollama": ("ollama_cloud_hint_label", "ollama_cloud_hint"),
        "groq": ("groq_quality_hint_label", "groq_quality_hint"),
    }

    @render.ui
    def loc_provider_hint():
        try:
            provider = input.provider_select()
        except Exception:
            return None
        keys = _PROVIDER_HINTS.get(provider)
        if not keys:
            return None
        label_key, text_key = keys
        return ui.tooltip(
            ui.tags.span(
                icon_svg("circle-info"),
                " ", t("sidebar", label_key),
                class_="text-muted small",
                style="cursor: help;",
            ),
            t("sidebar", text_key),
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
