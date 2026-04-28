"""Ollama provider: local model client with model-specific context tuning."""
import json

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
    "kimi-k2.6:cloud":        {"num_ctx": 262144, "num_predict": 131072, "temperature": 0.2},
    "deepseek-v4-pro:cloud":  {"num_ctx": 1048576, "num_predict": 262144, "temperature": 0.2},
    "deepseek-v4-flash:cloud":{"num_ctx": 1048576, "num_predict": 262144, "temperature": 0.2},
    "glm-5.1:cloud":          {"num_ctx": 202752, "num_predict": 131072, "temperature": 0.2},
    "gemma4:31b-cloud":       {"num_ctx": 262144, "num_predict": 131072, "temperature": 0.2},
}


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
        caps = _OLLAMA_MODEL_CONFIGS.get(
            model,
            {"num_predict": 65536, "num_ctx": 131072, "temperature": 0.2},
        )
        input_tokens = (
            _count_tokens(system_prompt)
            + _count_tokens(structure_hint) * 2
            + _count_tokens(rendered_user)
        )
        predicted_output = max(8192, input_tokens)
        num_predict = min(caps["num_predict"], predicted_output)
        needed_ctx = int((input_tokens + num_predict) * 1.10)
        num_ctx = min(caps["num_ctx"], _bucket_ctx(needed_ctx))
        options = {
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "temperature": caps.get("temperature", 0.2),
        }
        print(
            f"[OLLAMA DEBUG] model={model} input_tokens={input_tokens} "
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
        stream = client.chat(model=model, messages=messages, stream=True, options=options)

        content_parts = []
        done_reason = None
        for chunk in stream:
            # chunk is an ollama ChatResponse; .message.content holds the delta
            try:
                delta = chunk.message.content
            except AttributeError:
                delta = None
            if delta:
                content_parts.append(delta)
            # Capture why the stream ended (stop, length, etc.)
            try:
                if getattr(chunk, "done", False):
                    done_reason = getattr(chunk, "done_reason", None)
            except Exception:
                pass
        content = "".join(content_parts)

        preview = content[:300].replace("\n", " ") if content else "<empty>"
        print(f"[OLLAMA DEBUG] model={model} done_reason={done_reason} content_len={len(content)} preview={preview}")

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


