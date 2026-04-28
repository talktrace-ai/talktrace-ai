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


        # Make the API call with structured output
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
            }
        )

        _cache_put(cache_key, response.output_text)
        return response.output_text

    except Exception as e:
        print(f"[ERROR] OpenAI API error: {e}")
        return json.dumps({"error": str(e)})


