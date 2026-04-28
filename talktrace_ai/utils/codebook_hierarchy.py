"""Code-Priorität / Hierarchie aus dem Codebuch ableiten.

Verwendung in make_qualitative_stats_df:
- Bei single-coding (Default) entscheidet die Priorität, welcher Code überlebt,
  wenn das Modell denselben Turn mit mehreren Codes belegt.
- Bei multi-coding sortiert die Priorität die Reihenfolge der zusammengefassten
  Codes (höchste Priorität zuerst).

Auflösungsregel (Vorrang von oben nach unten):
1. Priorisierungs-Zeile irgendwo im Codebuch, z.B. ``Priorisierung: A1 > B2 > C3``.
   Erkannte Codes bekommen Priorität 0..n-1, höchste zuerst.
2. Explizite Spalte ``Priorität`` / ``Priority`` (oder Lower-Case-Varianten).
   Niedrigere Zahl = höhere Priorität.
3. Position des Codes im Codebuch (0-basiert).

Codes, die das Modell halluziniert (nicht im Codebuch), bekommen einen sehr
hohen Wert und landen damit in der Sortierung ans Ende.
"""
from __future__ import annotations

import re

# Erkenne sowohl deutsche als auch englische Spaltennamen, jeweils Title-
# und Lower-Case. Reihenfolge ist Look-up-Reihenfolge: ein expliziter Wert
# in irgendeiner der Schreibweisen genügt.
_PRIORITY_KEYS = ("Priorität", "Prioritaet", "Priority", "priorität", "prioritaet", "priority")
_CODE_KEYS = ("Code", "Shortcode", "code", "shortcode")
_UNKNOWN_PRIORITY = 1_000_000.0

# Erkennt eine optionale Label-Zeile vor der Code-Sequenz, z.B.
#   "Priorisierung: A1 > B2 > C3"
#   "Priority - A1 → B2 → C3"
#   "Hierarchie A1, B2, C3"
# Auch ohne Label akzeptiert: einfach "A1 > B2 > C3".
_PRIORITY_LABEL_RE = re.compile(
    r"^\s*(?:Priorisierung|Prioritaet|Priorität|Priorisation|"
    r"Hierarchie|Hierarchy|Rangfolge|Reihenfolge|Priority|Order)\s*[:.\-–—]?\s*",
    re.IGNORECASE,
)
# Trennzeichen zwischen Codes in einer Priorisierungs-Zeile.
_PRIORITY_SPLITTER_RE = re.compile(r"\s*(?:>|→|»|,|;|\|)\s*")


def _try_float(value):
    """Convert value to float; return None on failure or empty string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _extract_code(entry):
    """Pull the code identifier out of a codebook entry dict."""
    if not isinstance(entry, dict):
        return ""
    for key in _CODE_KEYS:
        if key in entry and entry[key]:
            return str(entry[key]).strip()
    # Fallback: erste nicht-leere Wert (Codebuch hat oft ein Layout
    # {"Code": "Q1", "Bezeichnung": "..."}, aber falls die erste Spalte
    # anders heißt, nehmen wir sie trotzdem)
    for v in entry.values():
        if v not in (None, ""):
            return str(v).strip()
    return ""


def _explicit_priority(entry):
    """Return the explicit priority value for an entry, if any column has one."""
    if not isinstance(entry, dict):
        return None
    for key in _PRIORITY_KEYS:
        if key in entry:
            v = _try_float(entry[key])
            if v is not None:
                return v
    return None


def extract_priority_line(codebook):
    """Suche nach einer Priorisierungs-Zeile in beliebigen Codebuch-Feldern.

    Akzeptierte Formen (Beispiele):
        "Priorisierung: A1 > B2 > C3"
        "Priority: A1 → B2 → C3"
        "Hierarchie - A1, B2, C3"
        "A1 > B2 > C3"  (auch ohne Label-Wort)

    Die Zeile gilt als gefunden, wenn nach optionalem Label mindestens **zwei**
    Tokens da sind, die als Codes im Codebuch tatsächlich existieren. Tokens,
    die kein bekannter Code sind, werden still ignoriert.

    Rückgabe: Liste der gefundenen Codes in Priorität-Reihenfolge (höchste
    zuerst), oder ``None`` wenn nichts erkannt wurde.
    """
    if not isinstance(codebook, list):
        return None
    known_codes = set()
    for entry in codebook:
        code = _extract_code(entry)
        if code:
            known_codes.add(code)
    if len(known_codes) < 2:
        return None
    for entry in codebook:
        if not isinstance(entry, dict):
            continue
        for value in entry.values():
            if not isinstance(value, str):
                continue
            ordered = _try_parse_priority_line(value, known_codes)
            if ordered:
                return ordered
    return None


def _try_parse_priority_line(text, known_codes):
    """Try to parse a priority sequence from a single string. Returns list or None."""
    if not text or not text.strip():
        return None
    cleaned = _PRIORITY_LABEL_RE.sub("", text.strip(), count=1).strip()
    if not cleaned:
        return None
    parts = [p.strip() for p in _PRIORITY_SPLITTER_RE.split(cleaned) if p.strip()]
    if len(parts) < 2:
        return None
    matches = [p for p in parts if p in known_codes]
    if len(matches) < 2:
        return None
    # Reihenfolge bewahren, aber Duplikate filtern (für den Fall, dass der
    # Nutzer denselben Code aus Versehen zweimal nennt).
    seen = set()
    ordered = []
    for c in matches:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def build_priority_lookup(codebook):
    """Return ``{shortcode: priority_float}`` for the given codebook.

    A lower number means higher priority (sorts first). Resolution order:

      1. **Priority line** ("Priorisierung: A1 > B2 > C3"): listed codes get
         priorities 0..n-1. Codes not mentioned fall through to step 2/3.
      2. **Explicit ``Priorität`` / ``Priority`` column**: numeric value used
         directly (offset by the number of codes already pinned by the line,
         so line-codes always sort first).
      3. **Position in codebook** (0-based, also offset by line length).

    Mixed mode is supported: codes named in the line get the line's order,
    codes with an explicit priority value get that value (offset), the rest
    fall back to position.
    """
    if not isinstance(codebook, list):
        return {}

    line_order = extract_priority_line(codebook)
    lookup = {}
    if line_order:
        for i, code in enumerate(line_order):
            lookup[code] = float(i)
    offset = len(line_order) if line_order else 0

    for i, entry in enumerate(codebook):
        code = _extract_code(entry)
        if not code or code in lookup:
            continue
        explicit = _explicit_priority(entry)
        if explicit is not None:
            lookup[code] = float(offset) + explicit
        else:
            lookup[code] = float(offset + i)
    return lookup


def priority_for(lookup, shortcode, default=None):
    """Return the priority value for a shortcode, or ``default`` when unknown.

    Codes the model hallucinated (not in lookup) get a very high value so
    they sort to the end. ``default`` overrides this if provided.
    """
    if shortcode in lookup:
        return lookup[shortcode]
    return _UNKNOWN_PRIORITY if default is None else default
