"""Ollama provider: local model client with model-specific context tuning."""
import json
import re
import time

from ollama import Client as OllamaClient, ResponseError as OllamaResponseError

from ..llm_cache import _cache_key, _cache_get, _cache_put
from ._json import (
    _format_codebook,
    _extract_json,
    _extract_items_by_schema,
    _extract_items_progressive,
    _repair_truncated_analysis,
)
from ._tokens import _count_tokens, _bucket_ctx


_OLLAMA_MODEL_CONFIGS = {
    # All current Ollama :cloud models emit chain-of-thought tokens before the
    # final JSON answer (observed: kimi-k2.6 ~25k, glm-5.1 ~30k). Reasoning
    # models need a larger num_predict floor so the generation budget covers
    # reasoning AND the JSON output. Local models (no :cloud suffix) are
    # assumed non-reasoning unless explicitly flagged.
    "kimi-k2.6:cloud":        {"num_ctx": 262144,  "num_predict": 131072, "temperature": 0.2, "reasoning": True},
    "deepseek-v4-pro:cloud":  {"num_ctx": 1048576, "num_predict": 262144, "temperature": 0.2, "reasoning": True},
    "deepseek-v4-flash:cloud":{"num_ctx": 1048576, "num_predict": 262144, "temperature": 0.2, "reasoning": True},
    "glm-5.1:cloud":          {"num_ctx": 202752,  "num_predict": 131072, "temperature": 0.2, "reasoning": True},
    "gemma4:31b-cloud":       {"num_ctx": 262144,  "num_predict": 131072, "temperature": 0.2, "reasoning": True},
}


def _caps_for_model(model):
    """Resolve per-model caps. Unknown :cloud models get the reasoning default
    (Ollama cloud currently ships only thinking-enabled models); local models
    fall back to a conservative non-reasoning default."""
    cfg = _OLLAMA_MODEL_CONFIGS.get(model)
    if cfg is not None:
        return cfg
    if model.endswith(":cloud"):
        return {"num_ctx": 262144, "num_predict": 131072, "temperature": 0.2, "reasoning": True}
    return {"num_ctx": 131072, "num_predict": 65536, "temperature": 0.2}


def llm_analysis_ollama(system_prompt, user_prompt, model, transcript, codebook, api_key=None):
    try:
        # Strong structural guidance so the model returns the exact shape we expect.
        # kimi-k2.5:cloud (and other Ollama models) have no strict schema support
        # like OpenAI, so we spell out the wrapper explicitly to avoid bare arrays
        # or empty responses.
        structure_hint = (
            "\n\nWICHTIG: Antworte AUSSCHLIESSLICH mit einem JSON-Objekt im folgenden Format "
            "(keine Markdown-Fences, kein zusätzlicher Text):\n"
            '{\n  "analysis": [\n    {"#": 1, "Sprecher": "...", "Shortcode": "...", "Impuls": "..."},\n'
            '    {"#": 2, "Sprecher": "...", "Shortcode": "...", "Impuls": "..."}\n  ]\n}\n'
            "Füge für JEDE codierbare Äußerung ALLER Sprecher:innen im Transkript einen Eintrag hinzu "
            "(Lehrperson UND Schüler:innen). Das Feld 'Sprecher' enthält die Kennung aus dem Transkript "
            "(z.B. 'Lehrperson', 'LEHRER', 'S01', 'S02', ...). "
            "Gib dir Mühe, tatsächlich Codes zuzuweisen — ein leeres Array ist fast immer falsch, "
            "weil reale Unterrichtsgespräche praktisch immer codierbare Beiträge enthalten. "
            "Ordne im Zweifelsfall den bestpassenden Code zu."
        )
        # Ankerbeispiele referenzieren oft ein anderes Nummerierungsschema
        # (z.B. T1-T21) als das aktuelle Transkript (S01, Lehrperson, ...).
        # Strikte JSON-Modelle wie kimi-k2:1t-cloud lesen die Ankerbeispiele
        # dann als Filter und geben ein leeres Analyse-Array zurück. Wir
        # entfernen die Spalte für die Ollama-Pipeline.
        codebook_for_ollama = codebook
        if isinstance(codebook, list):
            codebook_for_ollama = [
                {k: v for k, v in entry.items() if k != "Ankerbeispiel"}
                if isinstance(entry, dict) else entry
                for entry in codebook
            ]
        rendered_user = user_prompt.replace("{transcript}", str(transcript)).replace("{codebook}", _format_codebook(codebook_for_ollama))
        messages = [
            {"role": "system", "content": system_prompt + structure_hint},
            {"role": "user", "content": rendered_user + structure_hint},
        ]
        # Stream the response. Streaming keeps the TCP connection active,
        # so Cloudflare's 100s idle-timeout (the source of 524 errors for
        # slow models like kimi-k2.5:cloud coding every turn) no longer
        # fires. We accumulate the chunks into the final JSON string.
        # Generous output budget so the model doesn't truncate mid-JSON when
        # coding every turn of a long transcript.
        # Reaktive num_ctx/num_predict-Berechnung: skaliere mit der tatsächlichen
        # Input-Größe, gedeckelt durch die Pro-Modell-Maxima in _OLLAMA_MODEL_CONFIGS.
        # Spart KV-Cache (lokal) und vermeidet unnötig große num_predict-Budgets
        # (cloud, vermindert 524-Timeout-Wahrscheinlichkeit).
        caps = _caps_for_model(model)
        input_tokens = (
            _count_tokens(system_prompt)
            + _count_tokens(structure_hint) * 2
            + _count_tokens(rendered_user)
        )
        # Output budget: room for the JSON output (scales with input) plus,
        # for reasoning models, a fixed ~32k overhead for chain-of-thought.
        # Floor of 16k handles short transcripts cleanly.
        reasoning_overhead = 32768 if caps.get("reasoning") else 0
        predicted_output = max(16384, input_tokens + reasoning_overhead)
        num_predict = min(caps["num_predict"], predicted_output)
        needed_ctx = int((input_tokens + num_predict) * 1.10)
        num_ctx = min(caps["num_ctx"], _bucket_ctx(needed_ctx))
        options = {
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "temperature": caps.get("temperature", 0.2),
        }
        reasoning_tag = " reasoning" if caps.get("reasoning") else ""
        print(
            f"[OLLAMA DEBUG] model={model}{reasoning_tag} input_tokens={input_tokens} "
            f"num_ctx={num_ctx} num_predict={num_predict} "
            f"(caps={caps['num_ctx']}/{caps['num_predict']})"
        )

        # NOTE: do NOT pass `format="json"` here. Ollama's constrained-JSON
        # decoding mode causes trillion-param cloud models (e.g. kimi-k2:1t-cloud)
        # to collapse into the trivial `{"analysis": []}` output for any codebook
        # beyond the trivial case. We instead rely on the structure_hint in the
        # prompt plus the downstream _repair_truncated_analysis / _extract_json
        # path to extract JSON from free-form text. OpenAI still gets strict
        # schema enforcement via response_format=json_schema in its own function.
        # Local mode: always use local Ollama server at http://localhost:11434
        client = OllamaClient(host="http://localhost:11434")
        wall_start = time.monotonic()
        stream = client.chat(model=model, messages=messages, stream=True, options=options)

        content_parts = []
        thinking_parts = []
        done_reason = None
        # Final-chunk timing stats (Ollama returns these on the done=True chunk):
        # eval_count / eval_duration  -> output (generation) tokens & ns
        # prompt_eval_count / prompt_eval_duration -> input (prefill) tokens & ns
        timing = {}
        for chunk in stream:
            # chunk is an ollama ChatResponse; .message.content holds the delta.
            # Reasoning models (e.g. kimi-k2.6:cloud) emit the answer in
            # .message.thinking instead — capture both and prefer content.
            msg = getattr(chunk, "message", None)
            delta = getattr(msg, "content", None) if msg is not None else None
            thinking_delta = getattr(msg, "thinking", None) if msg is not None else None
            if delta:
                content_parts.append(delta)
            if thinking_delta:
                thinking_parts.append(thinking_delta)
            # Capture why the stream ended (stop, length, etc.) and timing stats.
            try:
                if getattr(chunk, "done", False):
                    done_reason = getattr(chunk, "done_reason", None)
                    for key in ("eval_count", "eval_duration",
                                "prompt_eval_count", "prompt_eval_duration"):
                        v = getattr(chunk, key, None)
                        if v is not None:
                            timing[key] = v
            except Exception:
                pass
        wall_elapsed = time.monotonic() - wall_start
        content = "".join(content_parts)
        thinking = "".join(thinking_parts)

        # Some cloud models put the JSON answer into the thinking channel and
        # leave content empty. Fall back to thinking when content is missing.
        if not content and thinking:
            content = thinking

        # Build the timing suffix. eval_duration is in nanoseconds.
        gen_info = ""
        ec, ed = timing.get("eval_count"), timing.get("eval_duration")
        if ec and ed:
            gen_rate = ec / (ed / 1e9)
            gen_info = f" gen={ec}tok @ {gen_rate:.1f} tok/s"
        prompt_info = ""
        pc, pd = timing.get("prompt_eval_count"), timing.get("prompt_eval_duration")
        if pc and pd:
            prompt_rate = pc / (pd / 1e9)
            prompt_info = f" prompt={pc}tok @ {prompt_rate:.1f} tok/s"

        preview = content[:300].replace("\n", " ") if content else "<empty>"
        print(
            f"[OLLAMA DEBUG] model={model} done_reason={done_reason} "
            f"content_len={len(content)} thinking_len={len(thinking)} "
            f"wall={wall_elapsed:.1f}s{gen_info}{prompt_info} preview={preview}"
        )

        if not content:
            return json.dumps({"error": "Ollama returned an empty response. Try a different model or retry."})

        # Strip markdown fences before parsing (cloud models often wrap JSON in ```json...```)
        cleaned = re.sub(r'```(?:json)?\s*', '', content).strip()
        cleaned = re.sub(r'```\s*$', '', cleaned).strip()

        def _count_items(s):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    arr = obj.get("analysis", [])
                    return len(arr) if isinstance(arr, list) else -1
                if isinstance(obj, list):
                    return len(obj)
            except Exception:
                return -1
            return -1

        # Try cleaned content first
        try:
            json.loads(cleaned)
            print(f"[OLLAMA DEBUG] parsed cleaned JSON, items={_count_items(cleaned)}")
            return cleaned
        except (json.JSONDecodeError, ValueError):
            pass

        # Try original content
        try:
            json.loads(content)
            return content
        except (json.JSONDecodeError, ValueError):
            pass

        # Try repairing a truncated JSON object (most common when done_reason
        # is "length" — the model hit num_predict mid-string). We walk back
        # to the last complete item of the "analysis" array and close braces.
        repaired = _repair_truncated_analysis(cleaned or content)
        if repaired:
            return repaired

        # Last resort: extract JSON object/array from the text
        extracted = _extract_json(content)
        if extracted:
            return extracted

        # Surface a bit of context so the user can see WHY parsing failed.
        preview = content[:160].replace("\n", " ")
        suffix = f" (done_reason={done_reason})" if done_reason else ""
        return json.dumps({"error": f"Failed to parse JSON from Ollama{suffix}. First 160 chars: {preview}"})

    except OllamaResponseError as e:
        msg = str(e)
        # Cloudflare 524: upstream (Ollama cloud) took too long. Provide a
        # concise, user-facing message instead of the full HTML error page.
        if "524" in msg or "timeout occurred" in msg.lower():
            return json.dumps({"error": "Ollama cloud timeout (524). The model took too long to respond. Try a smaller transcript, a different model, or retry in a few minutes."})
        return json.dumps({"error": f"Ollama error: {msg}"})

    except ConnectionError as e:
        return json.dumps({"error": "Cannot connect to Ollama. Make sure Ollama is running (local) or your API key is valid (cloud)."})

    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


