"""Incremental JSON parsing helpers for streaming providers.

The classic _extract_items_progressive walks top-level {...} blocks, which
is wrong for the streamed payload `{"analysis": [{...}, {...}]}` — it would
treat the outer object as a single candidate instead of yielding inner items.

`extract_new_items` walks INSIDE a named array key and tracks the resume
position between calls, so a streaming consumer can call it after each
delta and receive only the items completed since the previous call.

`parse_jsonl_line` validates a single line as a coded item and returns
the dict (or None on failure). Used by Groq/Ollama streaming.
"""
import json
import re


_REQUIRED_FIELDS = ("#", "Sprecher", "Shortcode", "Impuls")
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*$")


def find_array_start(buffer: str, array_key: str) -> int:
    """Return the index immediately after the opening `[` of the named array,
    or -1 if not yet present in the buffer.
    """
    key_token = f'"{array_key}"'
    key_idx = buffer.find(key_token)
    if key_idx < 0:
        return -1
    bracket_idx = buffer.find('[', key_idx + len(key_token))
    if bracket_idx < 0:
        return -1
    return bracket_idx + 1


def extract_new_items(buffer: str, after_pos: int):
    """Walk `buffer` from `after_pos` looking for complete inner objects.

    Returns (items, next_after_pos). `next_after_pos` advances past the
    last fully-parsed `}` (or stays put if a partial item is in flight).
    Items that fail to parse are skipped silently — they will retry on the
    next call once more characters arrive.
    """
    items = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    last_complete_end = after_pos
    i = after_pos
    n = len(buffer)
    while i < n:
        c = buffer[i]
        if in_str:
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = buffer[start:i + 1]
                parsed = _try_parse_object(candidate)
                if parsed is not None:
                    items.append(parsed)
                start = -1
                last_complete_end = i + 1
            elif depth < 0:
                # Stray '}' — likely the closing of the outer object after
                # the array. Stop walking; nothing more to extract.
                return items, last_complete_end
        elif c == ']' and depth == 0:
            # Array closed cleanly.
            return items, i + 1
        i += 1
    return items, last_complete_end


def _try_parse_object(candidate: str):
    """Parse a single object string; tolerate raw control chars (strict=False)."""
    for kwargs in ({}, {"strict": False}):
        try:
            obj = json.loads(candidate, **kwargs)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def normalize_item(obj):
    """Validate and normalize a coded item.

    Returns a dict with all four required fields (back-filling Sprecher to
    "" if missing, mirroring the classic path) or None if the object is
    not usable (missing Shortcode/Impuls).
    """
    if not isinstance(obj, dict):
        return None
    shortcode = obj.get("Shortcode")
    impuls = obj.get("Impuls")
    if shortcode is None or impuls is None:
        return None
    out = {
        "#": obj.get("#", 0),
        "Sprecher": obj.get("Sprecher", ""),
        "Shortcode": str(shortcode),
        "Impuls": str(impuls),
    }
    # Coerce # to int when possible; leave as-is otherwise.
    try:
        out["#"] = int(out["#"])
    except (TypeError, ValueError):
        pass
    return out


def strip_fence_line(line: str) -> str:
    """Return '' if the line is a markdown fence (``` or ```json), else
    the original line. Used to filter JSONL streams from Groq/Ollama."""
    if _FENCE_RE.match(line):
        return ""
    return line


def parse_jsonl_line(line: str):
    """Parse a single JSONL line into a normalized item, or None on failure.

    Empty lines and markdown fences return None. Lines that are not valid
    JSON or that miss required fields also return None — callers should
    log/discard rather than fail the whole stream.
    """
    text = line.strip()
    if not text:
        return None
    if _FENCE_RE.match(text):
        return None
    # Trim a trailing comma that some models emit by mistake.
    if text.endswith(","):
        text = text[:-1].rstrip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            obj = json.loads(text, strict=False)
        except (json.JSONDecodeError, ValueError):
            return None
    return normalize_item(obj)
