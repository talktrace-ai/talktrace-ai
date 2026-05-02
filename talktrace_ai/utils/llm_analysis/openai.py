"""OpenAI provider: Responses API with json_schema response_format.

Structured outputs: Wir bauen das Schema dynamisch aus dem Codebuch
(Shortcode-enum) und dem Transkript (Sprecher-enum). OpenAI's `strict: True`
macht das zur harten Constraint — das Modell kann gar keinen Code emittieren,
der nicht im Codebuch steht. Bei BadRequestError (z.B. zu großes enum für
ein Modell) fallen wir auf das alte unconstrained Schema zurück.
"""
import json

from openai import OpenAI, BadRequestError

from ..llm_cache import _cache_key, _cache_get, _cache_put
from ._json import _format_codebook
from ._schema import build_analysis_schema, has_enum_constraints
from ._stream_parse import (
    find_array_start,
    extract_new_items,
    normalize_item,
)


# Fallback-Schema ohne enum: identisch zum vor-Structured-Outputs-Stand, dient
# als Sicherheitsnetz, falls OpenAI das enum-Schema ablehnt (BadRequest).
_OPENAI_SCHEMA_NO_ENUM = {
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

        # Schema mit enum-Constraints aus Codebuch + Transkript bauen.
        # Bei strict=True erzwingt OpenAI die enum-Werte schon decoder-seitig.
        schema = build_analysis_schema(codebook, transcript)
        print(
            f"[OPENAI DEBUG] structured-outputs: enum_active={has_enum_constraints(schema)} model={model}"
        )

        rendered_user = (
            user_prompt.replace("{transcript}", str(transcript))
                       .replace("{codebook}", _format_codebook(codebook))
        )
        request_input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rendered_user},
        ]

        # Make the API call with structured output.
        # max_output_tokens explizit hochsetzen: ohne Cap fällt die Responses-API
        # auf den Modell-Default (~4-8k bei gpt-5er) und schneidet bei langen
        # Transkripten — speziell im Multi-Coding-Modus, wo pro Turn mehrere
        # Items emittiert werden — die Item-Liste mitten in einem JSON-Objekt ab.
        # 32k ist generös genug für ein typisches Klassengespräch (~24-100 Turns)
        # mit Multi-Coding und liegt unter den per-Modell-Limits aller gpt-5er.
        try:
            response = client.responses.create(
                model=model,
                input=request_input,
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
        except BadRequestError as e:
            # Schema rejected (z.B. zu großes enum, Modell unterstützt strict nicht).
            # Sauber auf das unconstrained Schema zurückfallen, statt abzustürzen.
            print(
                f"[OPENAI DEBUG] structured-outputs schema rejected ({e}); "
                f"falling back to unconstrained schema."
            )
            response = client.responses.create(
                model=model,
                input=request_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "analysis",
                        "schema": _OPENAI_SCHEMA_NO_ENUM,
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


def llm_analysis_openai_stream(
    system_prompt: str,
    user_prompt: str,
    model: str,
    transcript,
    codebook,
    client: OpenAI,
    _cancel_token=None,
):
    """Sync generator yielding {"type": "item"|"done"|"error"|"cancelled", ...} events.

    Uses the same strict json_schema response_format as the classic variant
    so the schema guarantee is preserved. Text deltas are accumulated and
    walked with extract_new_items to surface inner array elements as soon
    as they finish. If `_cancel_token` is provided and fires, iteration
    stops cleanly between events and a single "cancelled" event is yielded.
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
        # Schema mit enum-Constraints; bei BadRequest fallen wir auf das
        # unconstrained Schema zurück (gleiche Logik wie im Klassik-Pfad).
        schema = build_analysis_schema(codebook, transcript)
        print(
            f"[OPENAI STREAM] structured-outputs: enum_active={has_enum_constraints(schema)} model={model}"
        )

        # max_output_tokens: gleiche Begründung wie im Klassik-Pfad. Multi-Coding
        # emittiert pro Turn mehrere Items; ohne Cap bricht das Modell mitten in
        # der Liste ab (gpt-5er Default ~4-8k). 32k passt für ein typisches
        # Klassengespräch mit Multi-Coding und liegt unter den Per-Modell-Limits.
        def _build_kwargs(use_schema):
            return dict(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rendered_user},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "analysis",
                        "schema": use_schema,
                        "strict": True,
                    }
                },
                max_output_tokens=32000,
                stream=True,
            )

        request_kwargs = _build_kwargs(schema)

        partial_buffer = ""
        next_pos = 0
        array_start = -1
        emitted_count = 0
        final_text = ""

        try:
            event_iter = client.responses.create(**request_kwargs)
        except BadRequestError as e:
            print(
                f"[OPENAI STREAM] schema rejected ({e}); falling back to unconstrained schema."
            )
            event_iter = client.responses.create(**_build_kwargs(_OPENAI_SCHEMA_NO_ENUM))

        for event in event_iter:
            if _cancel_token is not None and _cancel_token.is_cancelled():
                print(f"[OPENAI STREAM] cancelled by user after {emitted_count} items")
                yield {"type": "cancelled", "items_so_far": emitted_count}
                return
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
