"""Anthropic provider: streaming completions with progressive JSON repair."""
import json

import anthropic as anthropic_sdk

from ..llm_cache import _cache_key, _cache_get, _cache_put
from ._json import (
    _format_codebook,
    _extract_json,
    _extract_items_by_schema,
    _extract_items_progressive,
    _repair_truncated_analysis,
)
from ._schema import extract_shortcodes, extract_speakers
from ._stream_parse import (
    find_array_start,
    extract_new_items,
    normalize_item,
)


def _strip_enums(tool):
    """Liefert eine Kopie des Analysis-Tools ohne enum-Constraints.

    Wird als Fallback genutzt, wenn Anthropic das enum-bestückte Schema mit
    einem BadRequestError ablehnt (z.B. wegen Schema-Größe).
    """
    import copy
    t = copy.deepcopy(tool)
    try:
        items = t["input_schema"]["properties"]["analysis"]["items"]["properties"]
        items["Shortcode"].pop("enum", None)
        items["Sprecher"].pop("enum", None)
    except (KeyError, TypeError):
        pass
    return t


def _build_analysis_tool(codebook, transcript):
    """Baue das submit_analysis-Tool mit enum-Constraints für Shortcode + Sprecher.

    Anthropic akzeptiert JSON-Schema-Standardfelder (inkl. ``enum``) im
    ``input_schema``. Damit erzwingt der Tool-Caller schon decoder-seitig die
    Code- und Sprechermenge — Halluzinationen wie "XX1" oder "Schueler02"
    landen nicht mehr im DataFrame.
    """
    shortcodes = extract_shortcodes(codebook)
    speakers = extract_speakers(transcript)

    shortcode_field = {
        "type": "string",
        "description": "The matching shortcode from the codebook.",
    }
    if shortcodes:
        shortcode_field["enum"] = shortcodes

    sprecher_field = {
        "type": "string",
        "description": "Speaker label (e.g. 'Lehrperson', 'LEHRER', 'S01', 'S02').",
    }
    if speakers:
        sprecher_field["enum"] = speakers

    return {
        "name": "submit_analysis",
        "description": (
            "Submit the qualitative coding analysis of the classroom transcript. "
            "The 'analysis' argument MUST be a native JSON array of objects — "
            "NOT a JSON-encoded string. Each '#' field MUST be a native integer — "
            "not a string. Include one object per coded utterance from any speaker "
            "(teacher and students). Use ONLY shortcodes that appear in the codebook."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "array",
                    "description": "Native JSON array (NOT a string) of coded utterances. One object per codable utterance in the transcript.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "#": {"type": "integer", "description": "Sequential index starting at 1."},
                            "Sprecher": sprecher_field,
                            "Shortcode": shortcode_field,
                            "Impuls": {"type": "string", "description": "The verbatim utterance text."},
                        },
                        "required": ["#", "Sprecher", "Shortcode", "Impuls"],
                    },
                },
            },
            "required": ["analysis"],
        },
    }


def llm_analysis_anthropic(system_prompt, user_prompt, model, transcript, codebook, client, progress_cb=None):
    """Anthropic implementation using forced tool_use for structured output.

    Forced tool_use is Anthropic's recommended pattern for structured output —
    equivalent to OpenAI's `response_format=json_schema`. The model is required
    to call our `submit_analysis` tool, and we extract the typed input. This
    bypasses the prose/markdown/refusal issues smarter models (Sonnet 4.5,
    Opus 4.6) exhibit when asked to "just output JSON" via prompt instructions.

    Nutzt Prompt-Caching: System-Prompt + Codebook-Block werden mit
    cache_control markiert, sodass bei wiederholten Analysen mit gleichem
    Codebook nur das (wechselnde) Transkript neu berechnet wird.
    """
    cache_key = _cache_key("anthropic", model, system_prompt, user_prompt, transcript, codebook)
    cached = _cache_get(cache_key)
    if cached is not None:
        print(f"[ANTHROPIC DEBUG] cache=HIT key={cache_key[:8]} — returning cached result")
        return cached

    stop_reason = None
    try:
        # Split: alles AUSSER dem Transkript gehört in den cache-fähigen Block
        # (Codebook + Intro + Schluss-Instruktion). Das Transkript ist der einzige
        # volatile Teil. So profitiert jede Wiederholungsanalyse vom Prompt-Cache.
        codebook_str = _format_codebook(codebook)
        with_codebook = user_prompt.replace("{codebook}", codebook_str)
        if "{transcript}" in with_codebook:
            before, after = with_codebook.split("{transcript}", 1)
            intro_text = before
            trailing_text = after
        else:
            intro_text = with_codebook
            trailing_text = ""

        # Tool-Instruktion VORNE: das Modell sieht zuerst, dass per Tool geantwortet
        # wird, nicht per JSON im Text — das entschärft den Konflikt mit der
        # "als JSON ausgeben"-Zeile im user_prompt-Template.
        tool_instruction = (
            "WICHTIG: Antworte ausschließlich durch Aufruf des Tools 'submit_analysis'. "
            "Übergib darin das vollständige Codierungs-Array. Codiere JEDE Äußerung im Transkript, "
            "die zu einem Code aus dem Codebuch passt — sowohl Äußerungen der Lehrperson als "
            "auch der Schüler:innen. Ordne im Zweifelsfall den bestpassenden Code zu; sei nicht "
            "überkritisch. Ein leeres Array ist fast immer falsch.\n\n"
        )

        stable_block = tool_instruction + intro_text + trailing_text
        volatile_block = str(transcript)

        user_content = [
            {"type": "text", "text": stable_block, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": volatile_block},
        ]

        print(
            f"[ANTHROPIC DEBUG] cache=miss sizes: transcript={len(volatile_block)} "
            f"codebook={len(codebook_str)} system={len(system_prompt)} "
            f"stable={len(stable_block)} model={model}"
        )

        # Output-Cap pro Modell. Deutlich reduziert gegenüber früher (32k/16k),
        # da tool_use-Payloads praktisch nie so groß werden und hohe max_tokens
        # bei einigen Modellen die Latenz erhöhen.
        if "opus" in model.lower() or "sonnet" in model.lower():
            max_tok = 16000
        else:
            max_tok = 8000

        # Define the structured-output tool. The model MUST call this tool.
        # Description emphasises native array/integer types — Sonnet 4.x tends
        # to stringify nested arrays under forced tool_use otherwise.
        # Codebuch- und Sprecher-enums sind im Helper enthalten.
        analysis_tool = _build_analysis_tool(codebook, transcript)
        n_codes = len(analysis_tool["input_schema"]["properties"]["analysis"]
                      ["items"]["properties"]["Shortcode"].get("enum", []))
        n_speakers = len(analysis_tool["input_schema"]["properties"]["analysis"]
                         ["items"]["properties"]["Sprecher"].get("enum", []))
        print(f"[ANTHROPIC DEBUG] structured-outputs: shortcode_enum={n_codes} sprecher_enum={n_speakers}")

        # Streaming + forced tool_use. tool_choice forces the model to call our tool.
        final_msg = None
        system_blocks = [
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ]

        def _run_stream(use_tool):
            nonlocal_final = {"msg": None}
            with client.messages.stream(
                model=model,
                max_tokens=max_tok,
                system=system_blocks,
                tools=[use_tool],
                tool_choice={"type": "tool", "name": "submit_analysis"},
                messages=[{"role": "user", "content": user_content}],
            ) as stream:
                chunks_seen = 0
                for _ in stream:
                    chunks_seen += 1
                    if progress_cb and chunks_seen % 20 == 0:
                        try:
                            progress_cb(chunks_seen)
                        except Exception:
                            pass
                nonlocal_final["msg"] = stream.get_final_message()
            return nonlocal_final["msg"]

        try:
            final_msg = _run_stream(analysis_tool)
        except anthropic_sdk.BadRequestError as e:
            print(
                f"[ANTHROPIC DEBUG] enum-Schema rejected ({e}); retry without enums."
            )
            final_msg = _run_stream(_strip_enums(analysis_tool))

        stop_reason = getattr(final_msg, "stop_reason", None)

        # Extract the tool_use input from the response content blocks.
        tool_input = None
        block_types = []
        for block in final_msg.content:
            btype = getattr(block, "type", "")
            block_types.append(btype)
            if btype == "tool_use" and getattr(block, "name", "") == "submit_analysis":
                tool_input = getattr(block, "input", None)
                break

        # Typ des analysis-Feldes inspizieren — manche SDK-Versionen liefern
        # bei Streaming mit forced tool_use den partial_json als String statt
        # als geparstes Dict/List zurück. Ohne Recovery landet das im UI als
        # "0 coded items", weil app.py non-list-analysis auf [] zurücksetzt.
        analysis_field = None
        if isinstance(tool_input, dict):
            analysis_field = tool_input.get("analysis")
        analysis_type = type(analysis_field).__name__ if analysis_field is not None else "None"
        analysis_len = len(analysis_field) if hasattr(analysis_field, "__len__") else -1
        usage = getattr(final_msg, "usage", None)
        print(
            f"[ANTHROPIC DEBUG] model={model} stop_reason={stop_reason} blocks={block_types} "
            f"analysis_type={analysis_type} analysis_len={analysis_len} usage={usage}"
        )

        # Recovery: analysis ist ein String — versuche, ihn als JSON-Array zu parsen.
        # Claude 4.x Sonnet/Opus stringifiziert bei forced tool_use mit nested
        # array-schemas gelegentlich den Array-Wert. Wir fangen das progressiv ab.
        if isinstance(analysis_field, str):
            sample = analysis_field[:200].replace("\n", " ")
            print(f"[ANTHROPIC DEBUG] analysis is STRING, first 200 chars: {sample!r}")
            parsed_items = None
            # Versuch 1: strict JSON parse
            try:
                cand = json.loads(analysis_field)
                if isinstance(cand, list):
                    parsed_items = cand
                    print(f"[ANTHROPIC DEBUG] strict parse OK — {len(cand)} items")
            except (json.JSONDecodeError, ValueError) as e:
                # Versuch 2: lenient parse (erlaubt rohe Control-Chars in Strings).
                try:
                    cand = json.loads(analysis_field, strict=False)
                    if isinstance(cand, list):
                        parsed_items = cand
                        print(f"[ANTHROPIC DEBUG] lenient (strict=False) parse OK — {len(cand)} items")
                except (json.JSONDecodeError, ValueError) as e2:
                    # Kontext um die Fehlerstelle loggen (hilft bei Diagnose).
                    pos = getattr(e2, "pos", None) or getattr(e, "pos", 0)
                    around = analysis_field[max(0, pos - 40):pos + 40]
                    print(f"[ANTHROPIC DEBUG] parse errors — strict: {e} / lenient: {e2}")
                    print(f"[ANTHROPIC DEBUG] context around pos {pos}: {around!r}")
                    # Versuch 3: Items einzeln via Brace-Walking extrahieren.
                    prog = _extract_items_progressive(analysis_field)
                    # Versuch 4: Feldbasierte Extraktion — robust gegen
                    # unescapte ASCII-Quotes in Impuls-Texten (Brace-Walking
                    # geht nach dem ersten kaputten Item aus dem Tritt).
                    schema = _extract_items_by_schema(analysis_field)
                    print(f"[ANTHROPIC DEBUG] progressive={len(prog)} schema={len(schema)}")
                    # Das Verfahren mit mehr Items wählen.
                    parsed_items = schema if len(schema) >= len(prog) else prog

            if parsed_items is not None and len(parsed_items) > 0:
                tool_input["analysis"] = parsed_items

        # Empty-Dump wie vorher, nachdem evtl. Recovery gelaufen ist.
        if isinstance(tool_input, dict):
            final_analysis = tool_input.get("analysis")
            final_n = len(final_analysis) if isinstance(final_analysis, list) else -1
            if final_n == 0:
                dump = json.dumps(tool_input, ensure_ascii=False)[:500]
                print(f"[ANTHROPIC DEBUG] empty tool_input first 500: {dump}")

        if isinstance(tool_input, dict) and isinstance(tool_input.get("analysis"), list):
            result = json.dumps(tool_input, ensure_ascii=False)
            _cache_put(cache_key, result)
            return result

        # Fallback: model didn't use the tool (rare under forced tool_choice).
        # Pull any text blocks and try to parse them as JSON.
        text_fallback = "".join(
            getattr(b, "text", "") for b in final_msg.content if getattr(b, "type", "") == "text"
        )
        print(f"[ANTHROPIC DEBUG] no tool_use block; text fallback first 500: {text_fallback[:500]!r}")
        extracted = _extract_json(text_fallback) if text_fallback else None
        if extracted:
            return extracted

        return json.dumps({
            "error": f"Anthropic did not return a tool_use call (stop_reason={stop_reason}, blocks={block_types})."
        })

    except anthropic_sdk.AuthenticationError as e:
        print(f"[ERROR]AuthenticationError: {str(e)}")
        return json.dumps({"error": "Authentication failed - Check API key or access rights."})

    except anthropic_sdk.RateLimitError as e:
        print(f"[ERROR]RateLimitError: {str(e)}")
        return json.dumps({"error": "Rate limit exceeded - Too many requests."})

    except anthropic_sdk.BadRequestError as e:
        print(f"[ERROR]BadRequestError: {str(e)}")
        return json.dumps({"error": f"Bad request: {str(e)}"})

    except anthropic_sdk.APIError as e:
        print(f"[ERROR]APIError: {str(e)}")
        return json.dumps({"error": f"API error: {str(e)}"})

    except Exception as e:
        print(f"[ERROR]Unexpected error: {str(e)}")
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def _build_anthropic_request(system_prompt, user_prompt, model, transcript, codebook):
    """Shared request construction for both classic and streaming variants.

    Returns (kwargs, system_blocks, analysis_tool, max_tok). Mirrors the
    setup in llm_analysis_anthropic so streaming uses identical prompt
    caching, tool schema and max_tokens behaviour.
    """
    codebook_str = _format_codebook(codebook)
    with_codebook = user_prompt.replace("{codebook}", codebook_str)
    if "{transcript}" in with_codebook:
        before, after = with_codebook.split("{transcript}", 1)
        intro_text = before
        trailing_text = after
    else:
        intro_text = with_codebook
        trailing_text = ""

    tool_instruction = (
        "WICHTIG: Antworte ausschließlich durch Aufruf des Tools 'submit_analysis'. "
        "Übergib darin das vollständige Codierungs-Array. Codiere JEDE Äußerung im Transkript, "
        "die zu einem Code aus dem Codebuch passt — sowohl Äußerungen der Lehrperson als "
        "auch der Schüler:innen. Ordne im Zweifelsfall den bestpassenden Code zu; sei nicht "
        "überkritisch. Ein leeres Array ist fast immer falsch.\n\n"
    )

    stable_block = tool_instruction + intro_text + trailing_text
    volatile_block = str(transcript)

    user_content = [
        {"type": "text", "text": stable_block, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": volatile_block},
    ]

    if "opus" in model.lower() or "sonnet" in model.lower():
        max_tok = 16000
    else:
        max_tok = 8000

    analysis_tool = _build_analysis_tool(codebook, transcript)

    system_blocks = [
        {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
    ]

    kwargs = dict(
        model=model,
        max_tokens=max_tok,
        system=system_blocks,
        tools=[analysis_tool],
        tool_choice={"type": "tool", "name": "submit_analysis"},
        messages=[{"role": "user", "content": user_content}],
    )
    return kwargs, len(volatile_block), len(codebook_str), len(stable_block)


def llm_analysis_anthropic_stream(system_prompt, user_prompt, model, transcript, codebook, client, _cancel_token=None):
    """Sync generator yielding {"type": "item"|"done"|"error"|"cancelled", ...} events.

    Uses the same forced tool_use call as the classic variant so the strict
    schema guarantee is preserved. As partial_json deltas arrive we walk the
    accumulated buffer and emit each fully-closed inner item exactly once.
    Cache replay is supported transparently: on a cache hit we re-emit the
    cached items as a stream of item events followed by done. If
    `_cancel_token` is provided and fires, the SDK stream context is exited
    cleanly and a single "cancelled" event is yielded.
    """
    cache_key = _cache_key("anthropic", model, system_prompt, user_prompt, transcript, codebook)
    cached = _cache_get(cache_key)
    if cached is not None:
        print(f"[ANTHROPIC STREAM] cache=HIT key={cache_key[:8]} — replaying")
        yield from _replay_cached(cached)
        return

    try:
        kwargs, vol_len, cb_len, stable_len = _build_anthropic_request(
            system_prompt, user_prompt, model, transcript, codebook
        )
        print(
            f"[ANTHROPIC STREAM] cache=miss sizes: transcript={vol_len} codebook={cb_len} "
            f"system={len(system_prompt)} stable={stable_len} model={model}"
        )

        partial_buffer = ""
        array_start = -1
        next_pos = 0
        emitted_count = 0
        final_msg = None

        # Stream öffnen mit graceful fallback: wenn das enum-Schema beim
        # __enter__ einen BadRequest auslöst (z.B. Schema zu groß), strippen
        # wir die enums und versuchen es einmal erneut. Vor dem __enter__
        # wird noch nichts geyieldet, also ist der Retry kosten-/zustandsfrei.
        try:
            stream_ctx = client.messages.stream(**kwargs)
            stream = stream_ctx.__enter__()
        except anthropic_sdk.BadRequestError as e:
            print(f"[ANTHROPIC STREAM] enum-Schema rejected ({e}); retry without enums.")
            kwargs["tools"] = [_strip_enums(kwargs["tools"][0])]
            stream_ctx = client.messages.stream(**kwargs)
            stream = stream_ctx.__enter__()

        try:
            cancelled = False
            for event in stream:
                if _cancel_token is not None and _cancel_token.is_cancelled():
                    print(f"[ANTHROPIC STREAM] cancelled by user after {emitted_count} items")
                    cancelled = True
                    break
                # Anthropic SDK emits ContentBlockDeltaEvent with delta.type
                # == "input_json_delta" carrying delta.partial_json (string).
                delta = getattr(event, "delta", None)
                if delta is not None and getattr(delta, "type", "") == "input_json_delta":
                    chunk = getattr(delta, "partial_json", "") or ""
                    if chunk:
                        partial_buffer += chunk
                        # Locate the array start once it appears in the buffer.
                        if array_start < 0:
                            array_start = find_array_start(partial_buffer, "analysis")
                            if array_start >= 0:
                                next_pos = array_start
                        if array_start >= 0:
                            new_items, next_pos = extract_new_items(partial_buffer, next_pos)
                            for raw in new_items:
                                norm = normalize_item(raw)
                                if norm is None:
                                    continue
                                emitted_count += 1
                                if not norm.get("#"):
                                    norm["#"] = emitted_count
                                yield {"type": "item", "data": norm}
            if cancelled:
                stream_ctx.__exit__(None, None, None)
                yield {"type": "cancelled", "items_so_far": emitted_count}
                return
            final_msg = stream.get_final_message()
        finally:
            try:
                stream_ctx.__exit__(None, None, None)
            except Exception:
                pass

        stop_reason = getattr(final_msg, "stop_reason", None)

        # Recovery: if streaming yielded nothing (e.g. SDK didn't surface
        # input_json_delta, or model stringified the array), fall back to
        # the classic post-processing pipeline.
        tool_input = None
        for block in (getattr(final_msg, "content", None) or []):
            if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "") == "submit_analysis":
                tool_input = getattr(block, "input", None)
                break

        recovered_items = None
        if isinstance(tool_input, dict):
            analysis_field = tool_input.get("analysis")
            if isinstance(analysis_field, list) and emitted_count == 0:
                recovered_items = [normalize_item(x) for x in analysis_field]
                recovered_items = [x for x in recovered_items if x is not None]
            elif isinstance(analysis_field, str):
                # Stringified array fallback (Claude 4.x quirk under tool_use).
                parsed = None
                for kw in ({}, {"strict": False}):
                    try:
                        cand = json.loads(analysis_field, **kw)
                        if isinstance(cand, list):
                            parsed = cand
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
                if parsed is None:
                    parsed = _extract_items_by_schema(analysis_field) or _extract_items_progressive(analysis_field)
                if parsed and emitted_count == 0:
                    recovered_items = [normalize_item(x) for x in parsed if isinstance(x, dict)]
                    recovered_items = [x for x in recovered_items if x is not None]

        if recovered_items:
            for i, item in enumerate(recovered_items, 1):
                if not item.get("#"):
                    item["#"] = i
                yield {"type": "item", "data": item}
            emitted_count = len(recovered_items)

        if emitted_count == 0:
            yield {
                "type": "error",
                "message": f"Anthropic stream produced no items (stop_reason={stop_reason}).",
            }
            return

        # Build cache payload from emitted items.
        # We need to re-collect; emit a synthetic raw_json identical in shape
        # to the classic return value.
        # The simplest approach is to re-walk recovered_items or rebuild from
        # tool_input if present and a list.
        items_for_cache = []
        if recovered_items:
            items_for_cache = recovered_items
        elif isinstance(tool_input, dict) and isinstance(tool_input.get("analysis"), list):
            items_for_cache = [normalize_item(x) for x in tool_input["analysis"]]
            items_for_cache = [x for x in items_for_cache if x is not None]

        if items_for_cache:
            raw_json = json.dumps({"analysis": items_for_cache}, ensure_ascii=False)
            _cache_put(cache_key, raw_json)
        else:
            raw_json = ""  # Don't poison the cache on partial streaming output.

        yield {"type": "done", "raw_json": raw_json, "stop_reason": stop_reason}

    except anthropic_sdk.AuthenticationError as e:
        yield {"type": "error", "message": "Authentication failed - Check API key or access rights."}
    except anthropic_sdk.RateLimitError as e:
        yield {"type": "error", "message": "Rate limit exceeded - Too many requests."}
    except anthropic_sdk.BadRequestError as e:
        yield {"type": "error", "message": f"Bad request: {e}"}
    except anthropic_sdk.APIError as e:
        yield {"type": "error", "message": f"API error: {e}"}


def _replay_cached(cached_json):
    """Replay a cached `{"analysis": [...]}` payload as a stream of events.

    Allows the streaming consumer to see the same per-item flow on a cache
    hit as on a fresh API call, so the UI updates uniformly either way.
    """
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
