"""OpenAI provider: Responses API with json_object response_format."""
import json

from openai import OpenAI

from ..llm_cache import _cache_key, _cache_get, _cache_put
from ._json import _format_codebook
from ._stream_parse import (
    find_array_start,
    extract_new_items,
    normalize_item,
)


_OPENAI_SCHEMA = {
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


def llm_analysis_openai_stream(
    system_prompt: str,
    user_prompt: str,
    model: str,
    transcript,
    codebook,
    client: OpenAI,
):
    """Sync generator yielding {"type": "item"|"done"|"error", ...} events.

    Uses the same strict json_schema response_format as the classic variant
    so the schema guarantee is preserved. Text deltas are accumulated and
    walked with extract_new_items to surface inner array elements as soon
    as they finish.
    """
    cache_key = _cache_key("openai", model, system_prompt, user_prompt, transcript, codebook)
    cached = _cache_get(cache_key)
    if cached is not None:
        print(f"[OPENAI STREAM] cache=HIT key={cache_key[:8]} — replaying")
        yield from _replay_cached(cached)
        return

    try:
        rendered_user = (
            user_prompt.replace("{transcript}", str(transcript))
                       .replace("{codebook}", _format_codebook(codebook))
        )
        request_kwargs = dict(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rendered_user},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "analysis",
                    "schema": _OPENAI_SCHEMA,
                    "strict": True,
                }
            },
            stream=True,
        )

        partial_buffer = ""
        next_pos = 0
        array_start = -1
        emitted_count = 0
        final_text = ""

        for event in client.responses.create(**request_kwargs):
            event_type = getattr(event, "type", "") or ""
            # Accumulate text deltas. The Responses API streaming surface uses
            # `response.output_text.delta` for incremental output_text. Other
            # event types (created/in_progress/output_item.added/.done/etc.)
            # are ignored here.
            if event_type.endswith("output_text.delta"):
                chunk = getattr(event, "delta", "") or ""
                if chunk:
                    partial_buffer += chunk
            elif event_type.endswith("output_text.done"):
                final_text = getattr(event, "text", "") or final_text
            elif event_type.endswith("response.completed"):
                resp = getattr(event, "response", None)
                if resp is not None:
                    txt = getattr(resp, "output_text", None)
                    if txt:
                        final_text = txt
            else:
                continue

            if not partial_buffer:
                continue
            if array_start < 0:
                array_start = find_array_start(partial_buffer, "analysis")
                if array_start >= 0:
                    next_pos = array_start
                else:
                    continue
            new_items, next_pos = extract_new_items(partial_buffer, next_pos)
            for raw in new_items:
                norm = normalize_item(raw)
                if norm is None:
                    continue
                emitted_count += 1
                if not norm.get("#"):
                    norm["#"] = emitted_count
                yield {"type": "item", "data": norm}

        # Fall back to parsing the final text if streaming surfaced nothing.
        if emitted_count == 0:
            text_to_parse = final_text or partial_buffer
            try:
                obj = json.loads(text_to_parse)
            except (json.JSONDecodeError, ValueError):
                obj = None
            items = []
            if isinstance(obj, dict):
                arr = obj.get("analysis", [])
                if isinstance(arr, list):
                    items = [normalize_item(x) for x in arr]
                    items = [x for x in items if x is not None]
            for i, item in enumerate(items, 1):
                if not item.get("#"):
                    item["#"] = i
                yield {"type": "item", "data": item}
                emitted_count += 1

        if emitted_count == 0:
            yield {"type": "error", "message": "OpenAI stream produced no items."}
            return

        if final_text:
            _cache_put(cache_key, final_text)
        yield {"type": "done", "raw_json": final_text, "stop_reason": "completed"}

    except Exception as e:
        print(f"[ERROR] OpenAI stream error: {e}")
        yield {"type": "error", "message": str(e)}


def _replay_cached(cached_json):
    try:
        obj = json.loads(cached_json)
    except (json.JSONDecodeError, ValueError):
        yield {"type": "error", "message": "Cached payload was unparseable."}
        return
    items = obj.get("analysis", []) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
    for raw in items:
        norm = normalize_item(raw) if isinstance(raw, dict) else None
        if norm is not None:
            yield {"type": "item", "data": norm}
    yield {"type": "done", "raw_json": cached_json, "stop_reason": "cache_hit"}
