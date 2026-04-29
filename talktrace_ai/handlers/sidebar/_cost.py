"""Token counting + cost estimation + sidebar cost-prediction line."""
from .._common import *


def register(state):
    input = state.input
    t = state.t
    config = state.config
    model = state.model
    transcript_data = state.transcript_data
    codebook_data = state.codebook_data
    estimated_cost = state.estimated_cost
    token_count = state.token_count
    current_lang = state.current_lang

    def calculate_input_tokens(transcript, codebook, system_prompt_text, user_prompt_text):
        """Calculate approximate token count for LLM request"""
        try:
            # Use the encoding for the selected model
            if config.get_current_api() == "openai":
                try:
                    encoding = tiktoken.encoding_for_model(model.get())
                except:
                    encoding = tiktoken.get_encoding("cl100k_base")
            else:  # groq, anthropic, ollama
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

    def _format_cost(cost: float, lang: str) -> str:
        """0.00 (en) or 0,00 (de) — two decimals, locale-aware separator."""
        s = f"{cost:.2f}"
        return s.replace(".", ",") if lang == "de" else s

    @render.ui
    def cost_chip():
        # Defensive reads: at first render (before inputs are wired up) Shiny
        # raises SilentException for missing inputs; catching it lets the
        # placeholder chip appear instead of leaving the slot stuck.
        try:
            llm_on = bool(input.llm_switch())
        except Exception:
            llm_on = False
        try:
            lang = current_lang.get()
        except Exception:
            lang = "en"
        tokens = token_count.get()
        cost = estimated_cost.get()
        if llm_on and tokens and cost is not None:
            amount = f"≈ {_format_cost(cost, lang)} €"
        else:
            amount = f"{_format_cost(0.0, lang)} €"
        tooltip = t("sidebar", "cost_prediction")
        if tokens:
            tooltip = f"{tooltip} · {t('sidebar', 'tokens_aprox')} {tokens:,}"
        return ui.div(
            icon_svg("coins"),
            ui.span(amount, class_="ttai-cost-chip__amount"),
            class_="ttai-cost-chip",
            title=tooltip,
        )

    # Update cost prediction when transcript/codebook changes
    @reactive.effect
    def update_cost_prediction():
        req(transcript_data.get() != None, codebook_data.get() != None, input.llm_switch())
        tokens = calculate_input_tokens(
            transcript_data.get(),
            codebook_data.get() or "",
            state.effective_system_prompt(),
            state.effective_user_prompt()
        )
        token_count.set(tokens)
        cost = calculate_estimated_cost(tokens)
        estimated_cost.set(cost)
