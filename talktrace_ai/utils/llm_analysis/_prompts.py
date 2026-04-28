"""Prompt overrides for streaming variants.

Groq and Ollama in streaming mode use a JSONL output contract — one JSON
object per line, no wrapper, no array, no markdown fences. This format
allows incremental parsing line-by-line as deltas arrive.

OpenAI and Anthropic keep their structured-output schemas (response_format=
json_schema and forced tool_use respectively) and parse the streamed JSON
array progressively via _extract_items_progressive — no prompt change needed.
"""

JSONL_OVERRIDE_DE = (
    "\n\nWICHTIG: Antworte ausschliesslich im JSONL-Format. Gib für JEDE "
    "codierte Äußerung GENAU EIN JSON-Objekt auf einer eigenen Zeile aus. "
    "Keine Wrapper, keine umschließenden Arrays oder Objekte, keine Markdown-Fences, "
    "kein zusätzlicher Text. Eine Zeile = ein Objekt mit genau diesen Feldern:\n"
    '{"#": <int>, "Sprecher": "<...>", "Shortcode": "<...>", "Impuls": "<wörtlich>"}\n'
    "Codiere ALLE Sprecher:innen (Lehrperson UND Schüler:innen). Numeriere "
    "fortlaufend ab 1. Zitiere Impulse wörtlich, fasse nicht zusammen. "
    "Ein leeres Ergebnis ist fast immer falsch."
)

JSONL_OVERRIDE_EN = (
    "\n\nIMPORTANT: Respond strictly in JSONL format. For EACH coded utterance, "
    "emit EXACTLY ONE JSON object on its own line. No wrappers, no surrounding "
    "arrays or objects, no markdown fences, no extra text. One line = one object "
    "with exactly these fields:\n"
    '{"#": <int>, "Sprecher": "<...>", "Shortcode": "<...>", "Impuls": "<verbatim>"}\n'
    "Code ALL speakers (teacher AND students). Number sequentially starting at 1. "
    "Quote utterances verbatim, do not summarize. An empty result is almost always wrong."
)


def jsonl_override(language: str) -> str:
    """Return the JSONL output instruction for the given language."""
    if (language or "").lower().startswith("en"):
        return JSONL_OVERRIDE_EN
    return JSONL_OVERRIDE_DE
