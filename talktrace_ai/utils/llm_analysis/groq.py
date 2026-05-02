"""Groq provider: chat-completion call with response-format JSON.

To add a new provider, copy this file, rename the function to
``llm_analysis_<name>``, swap the SDK call, and re-export from
``__init__.py``. Provider-specific exceptions go in the except block.
"""
import json

from groq import BadRequestError, AuthenticationError, RateLimitError, InternalServerError, APIError

from ..llm_cache import _cache_key, _cache_get, _cache_put
from ._json import _format_codebook
from ._prompts import jsonl_override
from ._schema import build_analysis_schema, has_enum_constraints
from ._stream_parse import parse_jsonl_line


def llm_analysis_groq(system_prompt, user_prompt, model, transcript, codebook, client):
    cache_key = _cache_key("groq", model, system_prompt, user_prompt, transcript, codebook)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt.replace("{transcript}", str(transcript)).replace("{codebook}", _format_codebook(codebook)),
        },
    ]

    # Structured Outputs: Groq unterstützt response_format=json_schema für eine
    # wachsende Liste von Modellen (insbesondere kimi-k2, llama-3.3, openai/gpt-oss).
    # Wir versuchen es zuerst mit Schema (inkl. enum); bei BadRequest fällt der
    # Code auf das alte json_object-Format zurück, sodass auch ältere Modelle
    # weiterhin funktionieren.
    schema = build_analysis_schema(codebook, transcript)
    print(
        f"[GROQ DEBUG] structured-outputs: enum_active={has_enum_constraints(schema)} model={model}"
    )

    def _create(response_format):
        return client.chat.completions.create(
            messages=messages,
            model=model,
            response_format=response_format,
            max_tokens=12000,
        )

    try:
        try:
            chat_completion = _create({
                "type": "json_schema",
                "json_schema": {"name": "analysis", "schema": schema, "strict": True},
            })
        except BadRequestError as e:
            print(f"[GROQ DEBUG] json_schema rejected ({e}); falling back to json_object.")
            chat_completion = _create({"type": "json_object"})

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


def llm_analysis_groq_stream(system_prompt, user_prompt, model, transcript, codebook, client, language="de"):
    """Sync generator yielding {"type": "item"|"done"|"error", ...} events.

    Uses a JSONL output contract: the model is asked to emit one JSON object
    per line with no wrapper/array. We enable HTTP streaming and parse lines
    as they arrive. Each emitted item is validated; malformed lines are
    discarded silently.
    """
    cache_key = _cache_key("groq", model, system_prompt, user_prompt, transcript, codebook)
    cached = _cache_get(cache_key)
    if cached is not None:
        print(f"[GROQ STREAM] cache=HIT key={cache_key[:8]} — replaying")
        yield from _replay_cached(cached)
        return

    try:
        override = jsonl_override(language)
        rendered_user = (
            user_prompt.replace("{transcript}", str(transcript))
                       .replace("{codebook}", _format_codebook(codebook))
        )
        # Append the JSONL override to BOTH system and user prompts so the
        # instruction is impossible to miss. response_format is intentionally
        # NOT set — json_object would force a wrapper, which we don't want.
        messages = [
            {"role": "system", "content": system_prompt + override},
            {"role": "user", "content": rendered_user + override},
        ]
        stream = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=12000,
            stream=True,
        )

        line_buffer = ""
        items_for_cache = []
        emitted_count = 0
        for chunk in stream:
            try:
                choice = chunk.choices[0]
            except (IndexError, AttributeError):
                continue
            delta = getattr(choice, "delta", None)
            text = getattr(delta, "content", None) if delta is not None else None
            if not text:
                continue
            line_buffer += text
            while "\n" in line_buffer:
                line, line_buffer = line_buffer.split("\n", 1)
                item = parse_jsonl_line(line)
                if item is None:
                    continue
                emitted_count += 1
                if not item.get("#"):
                    item["#"] = emitted_count
                items_for_cache.append(item)
                yield {"type": "item", "data": item}

        # Drain trailing buffer (last line may have no terminating newline).
        if line_buffer.strip():
            item = parse_jsonl_line(line_buffer)
            if item is not None:
                emitted_count += 1
                if not item.get("#"):
                    item["#"] = emitted_count
                items_for_cache.append(item)
                yield {"type": "item", "data": item}

        if emitted_count == 0:
            yield {"type": "error", "message": "Groq stream produced no items."}
            return

        raw_json = json.dumps({"analysis": items_for_cache}, ensure_ascii=False)
        _cache_put(cache_key, raw_json)
        yield {"type": "done", "raw_json": raw_json, "stop_reason": "completed"}

    except BadRequestError as e:
        yield {"type": "error", "message": "Bad request - Failed to generate JSON."}
    except AuthenticationError as e:
        yield {"type": "error", "message": "Authentication failed - Check API key or access rights."}
    except RateLimitError as e:
        yield {"type": "error", "message": "Rate limit exceeded - Too many requests."}
    except InternalServerError as e:
        yield {"type": "error", "message": "Server error - Please try again later."}
    except APIError as e:
        yield {"type": "error", "message": f"API error: {e}"}
    except Exception as e:
        print(f"[ERROR] Groq stream unexpected: {e}")
        yield {"type": "error", "message": f"Unexpected error: {e}"}


def _replay_cached(cached_json):
    try:
        obj = json.loads(cached_json)
    except (json.JSONDecodeError, ValueError):
        yield {"type": "error", "message": "Cached payload was unparseable."}
        return
    from ._stream_parse import normalize_item
    items = obj.get("analysis", []) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
    for raw in items:
        norm = normalize_item(raw) if isinstance(raw, dict) else None
        if norm is not None:
            yield {"type": "item", "data": norm}
    yield {"type": "done", "raw_json": cached_json, "stop_reason": "cache_hit"}
