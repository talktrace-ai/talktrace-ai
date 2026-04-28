"""OpenAI provider: Responses API with json_object response_format."""
import json

from openai import OpenAI

from ..llm_cache import _cache_key, _cache_get, _cache_put
from ._json import _format_codebook


def llm_analysis_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    transcript,
    codebook,
    client: OpenAI
) -> str:
    cache_key = _cache_key("openai", model, system_prompt, user_prompt, transcript, codebook)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:

        # Define schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "#": {"type": "integer", "description": "Nummerierung"},
                            "Sprecher": {"type": "string", "description": "Sprecher-Kennung (z.B. 'Lehrperson', 'LEHRER', 'S01', ...)"},
                            "Shortcode": {"type": "string", "description": "Der Shortcode"},
                            "Impuls": {"type": "string", "description": "Die Äußerung"}
                        },
                        "required": ["#", "Sprecher", "Shortcode", "Impuls"],
                        "additionalProperties": False
                    },
                    "description": "Liste von Analyseobjekten"
                }
            },
            "required": ["analysis"],
            "additionalProperties": False
        }


        # Make the API call with structured output.
        # max_output_tokens explizit hochsetzen: ohne Cap fällt die Responses-API
        # auf den Modell-Default (~4-8k bei gpt-5er) und schneidet bei langen
        # Transkripten — speziell im Multi-Coding-Modus, wo pro Turn mehrere
        # Items emittiert werden — die Item-Liste mitten in einem JSON-Objekt ab.
        # 32k ist generös genug für ein typisches Klassengespräch (~24-100 Turns)
        # mit Multi-Coding und liegt unter den per-Modell-Limits aller gpt-5er.
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.replace("{transcript}", str(transcript)).replace("{codebook}", _format_codebook(codebook))}
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "analysis",
                    "schema": schema,
                    "strict": True
                }
            },
            max_output_tokens=32000,
        )

        # Truncation surface: die Responses-API liefert `status` und
        # `incomplete_details.reason` zurück, wenn die Antwort am Cap
        # abgeschnitten wurde. Ohne diesen Check würde das UI stillschweigend
        # weniger Items zeigen, als das Modell hätte produzieren wollen.
        status = getattr(response, "status", None)
        if status and status != "completed":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) if details else None
            print(
                f"[OPENAI DEBUG] response status={status} incomplete_reason={reason} "
                f"output_text_len={len(response.output_text or '')} model={model}"
            )

        _cache_put(cache_key, response.output_text)
        return response.output_text

    except Exception as e:
        print(f"[ERROR] OpenAI API error: {e}")
        return json.dumps({"error": str(e)})


