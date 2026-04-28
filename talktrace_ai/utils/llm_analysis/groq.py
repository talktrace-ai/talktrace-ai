"""Groq provider: chat-completion call with response-format JSON.

To add a new provider, copy this file, rename the function to
``llm_analysis_<name>``, swap the SDK call, and re-export from
``__init__.py``. Provider-specific exceptions go in the except block.
"""
import json

from groq import BadRequestError, AuthenticationError, RateLimitError, InternalServerError, APIError

from ..llm_cache import _cache_key, _cache_get, _cache_put
from ._json import _format_codebook


def llm_analysis_groq(system_prompt, user_prompt, model, transcript, codebook, client):
    cache_key = _cache_key("groq", model, system_prompt, user_prompt, transcript, codebook)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        # Create chat completion object with JSON response format
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt.replace("{transcript}", str(transcript)).replace("{codebook}", _format_codebook(codebook)),
                }
            ],
            model=model,
            response_format={"type": "json_object"},
            max_tokens=12000,
        )

        analysis_json_string = chat_completion.choices[0].message.content
        _cache_put(cache_key, analysis_json_string)
        return analysis_json_string
    
    except BadRequestError as e:
        print(f"[ERROR]BadRequestError (400): {str(e)}")
        return json.dumps({"error": "Bad request - Failed to generate JSON."})

    except AuthenticationError as e:
        print(f"[ERROR]AuthenticationError (403): {str(e)}")
        return json.dumps({"error": "Authentication failed - Check API key or access rights."})

    except RateLimitError as e:
        print(f"[ERROR]RateLimitError (429): {str(e)}")
        return json.dumps({"error": "Rate limit exceeded - Too many requests."})

    except InternalServerError as e:
        print(f"[ERROR]InternalServerError (500+): {str(e)}")
        return json.dumps({"error": "Server error - Please try again later."})

    except APIError as e:
        print(f"[ERROR]APIError: {str(e)}")
        return json.dumps({"error": f"API error: {str(e)}"})

    except Exception as e:
        print(f"[ERROR]Unexpected error: {str(e)}")
        return json.dumps({"error": f"Unexpected error: {str(e)}"})
    

