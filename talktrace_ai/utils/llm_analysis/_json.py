"""Provider-agnostic JSON extraction & repair helpers.

Used by every llm_analysis_* provider to turn the raw model output into a
structured DataFrame. No SDK imports here — keeps it cheap to load.
"""
import json
import re

import pandas as pd


def _extract_json(text):
    """Try to extract a valid JSON object or array from text that may contain markdown fences, prose, or extra content."""
    if not text:
        return None
    # Fast path: strip markdown code fences and try parsing as-is.
    stripped = re.sub(r'```(?:json)?\s*', '', text).strip()
    stripped = re.sub(r'```\s*$', '', stripped).strip()
    try:
        json.loads(stripped)
        return stripped
    except (json.JSONDecodeError, ValueError):
        pass

    # Brace-balanced scan: find each candidate '{' or '[' and try to parse
    # the substring from there to its matching close. This is resilient to
    # prose prefixes, markdown fences, trailing commentary, and multi-JSON
    # artifacts like "{```json\n{...}" that a naive regex mishandles.
    def _balanced_parse(s, open_ch, close_ch):
        for start in range(len(s)):
            if s[start] != open_ch:
                continue
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(s)):
                c = s[i]
                if in_str:
                    if escape:
                        escape = False
                    elif c == '\\':
                        escape = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c == open_ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = s[start:i + 1]
                        try:
                            parsed = json.loads(candidate)
                            # Prefer candidates that actually look like our schema.
                            if isinstance(parsed, dict) and "analysis" in parsed:
                                return candidate
                            if isinstance(parsed, list):
                                return candidate
                            # Keep looking for a better candidate; fall through.
                            return candidate
                        except (json.JSONDecodeError, ValueError):
                            break  # unbalanced / malformed; try next start
        return None

    result = _balanced_parse(text, '{', '}')
    if result:
        return result
    result = _balanced_parse(text, '[', ']')
    if result:
        # Wrap bare array in {"analysis": [...]}
        try:
            arr = json.loads(result)
            if isinstance(arr, list):
                return json.dumps({"analysis": arr})
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _format_codebook(codebook):
    """Format codebook for LLM consumption. Converts list of dicts to readable JSON instead of Python repr."""
    if isinstance(codebook, list):
        return json.dumps(codebook, ensure_ascii=False, indent=2)
    return str(codebook)


def _extract_items_by_schema(text):
    """Field-aware recovery. Matches the fixed {#, Sprecher, Shortcode, Impuls}
    schema via regex and extracts each item. Robust against unescaped ASCII
    quotes inside Impuls text (a common Claude 4.x stringification artifact) —
    those break generic JSON parsers but not field-pattern matching.
    """
    items = []
    # Match item prefix up to the Impuls value opening quote. Everything before
    # Impuls is short/simple enough that inner quotes are unlikely.
    prefix_re = re.compile(
        r'\{\s*'
        r'"#"\s*:\s*(\d+)\s*,\s*'
        r'"Sprecher"\s*:\s*"([^"]*)"\s*,\s*'
        r'"Shortcode"\s*:\s*"([^"]*)"\s*,\s*'
        r'"Impuls"\s*:\s*"',
        re.DOTALL,
    )
    # Terminator: a quote followed by optional whitespace + closing brace,
    # followed by comma, closing bracket, or end-of-string. This is the only
    # reliable way to find the real end of Impuls when it may contain ".
    term_re = re.compile(r'"\s*\}\s*(?=,|\]|$)')

    pos = 0
    while pos < len(text):
        m = prefix_re.search(text, pos)
        if not m:
            break
        num_str, sprecher, shortcode = m.group(1), m.group(2), m.group(3)
        impuls_start = m.end()
        t = term_re.search(text, impuls_start)
        if not t:
            # last item without trailing comma — try relaxed terminator.
            t2 = re.search(r'"\s*\}\s*$', text[impuls_start:])
            if t2:
                impuls_end = impuls_start + t2.start()
                next_pos = len(text)
            else:
                break
        else:
            impuls_end = t.start()
            next_pos = t.end()
        impuls = text[impuls_start:impuls_end]
        try:
            items.append({
                "#": int(num_str),
                "Sprecher": sprecher,
                "Shortcode": shortcode,
                "Impuls": impuls,
            })
        except ValueError:
            pass
        pos = next_pos
    return items


def _extract_items_progressive(text):
    """Walk a JSON-array-like string and extract each top-level {...} block
    as a parsed dict. Skips items that fail to parse (e.g. unescaped quote
    inside an Impuls string) — robust recovery when the whole-array parse
    fails partway through. Tries strict JSON first, then strict=False.
    """
    items = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    i = 0
    while i < len(text):
        c = text[i]
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
                candidate = text[start:i + 1]
                parsed = None
                for kwargs in ({}, {"strict": False}):
                    try:
                        parsed = json.loads(candidate, **kwargs)
                        break
                    except (json.JSONDecodeError, ValueError):
                        continue
                if isinstance(parsed, dict):
                    items.append(parsed)
                start = -1
        i += 1
    return items


def _repair_truncated_analysis(text):
    """Salvage a truncated JSON response of the form {"analysis": [...]}.

    When the model hits its output cap mid-array, the tail looks like:
        ..., {"#": 7, "Sprecher": "S01", "Shortcode": "EF", "Imp
    We walk back to the last complete '}' that closes an item, drop
    the dangling partial, and close the array + object. Returns the
    repaired JSON string, or None if repair isn't possible.
    """
    if not text:
        return None
    # Find the opening of the analysis array.
    arr_start = text.find('"analysis"')
    if arr_start < 0:
        return None
    bracket_start = text.find('[', arr_start)
    if bracket_start < 0:
        return None
    # Find the outer '{' that opens the object containing "analysis".
    # Walk backwards from arr_start to skip any markdown fence / prose prefix.
    obj_start = text.rfind('{', 0, arr_start)
    if obj_start < 0:
        return None
    # Walk the array, tracking brace depth, to find the last complete item.
    depth = 0
    last_complete = -1  # index just after the most recent fully-closed item
    in_str = False
    escape = False
    for i in range(bracket_start + 1, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                last_complete = i + 1
        elif c == ']' and depth == 0:
            # Array closed cleanly; let the normal parser handle it.
            return None
    if last_complete < 0:
        # Nothing complete — return {"analysis": []}.
        return json.dumps({"analysis": []})
    # Build: text[obj_start..last_complete] + ']' + '}' (closes array + outer object).
    # Slicing from obj_start strips any leading prose/markdown fence.
    repaired = text[obj_start:last_complete] + "]}"
    try:
        json.loads(repaired)
        return repaired
    except (json.JSONDecodeError, ValueError):
        return None


