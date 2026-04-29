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

    @render.text
    def loc_display_cost_prediction():
        req(transcript_data.get() != None, codebook_data.get() != None)
        if input.llm_switch():
            tokens = token_count.get()
            cost = estimated_cost.get()
            if tokens and cost:
                return f"{t("sidebar", "tokens_aprox")} {tokens:} {t("sidebar", "cost_prediction")}: {cost:.4f} €"
        return ""
