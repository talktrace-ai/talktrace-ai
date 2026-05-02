"""Schema-Builder für Structured Outputs mit enum-Constraints.

Bisher ließen alle vier Provider das Modell beliebige Strings für Shortcode
und Sprecher emittieren — Fehlcodes ("XX1" für "Q1") und Phantom-Sprecher
("Schueler" statt "S03") fielen erst bei der nachgelagerten Pivot-Auswertung
auf. Ein JSON-Schema-`enum` zwingt das Modell schon decoder-seitig in die
Menge der gültigen Werte und erspart uns Re-Mapping-Heuristiken.

Codebuch-Layout: list[dict] mit "Code"/"Shortcode"-Spalte (siehe
``codebook_hierarchy._CODE_KEYS``). Sprecher: aus dem Transkript via
"<Label>:"-Präfixregex am Zeilenanfang.
"""
from __future__ import annotations

import re
from typing import Iterable

from ..codebook_hierarchy import _extract_code


# Sprecher-Zeile: "LEHRER:", "S01:", "Lehrperson:" etc. am Zeilenanfang.
# Kein Anker auf Großbuchstaben, weil "Lehrperson" / "Schüler:innen" auch
# legitim sind. Doppelpunkt + Whitespace markiert die Grenze.
_SPEAKER_LINE_RE = re.compile(r"^\s*([^:\n]{1,40}?)\s*:\s", re.MULTILINE)

# Generischer Schüler-Fallback, falls das Transkript leer / unparsbar ist.
# Deckt typische Manuskript-Konventionen ab: S01-S40 + Lehrperson/Lehrer.
_FALLBACK_SPEAKERS_DE = (
    ["Lehrperson", "LEHRER", "Lehrer", "Lehrerin", "Schüler:innen"]
    + [f"S{i:02d}" for i in range(1, 41)]
)
_FALLBACK_SPEAKERS_EN = (
    ["Teacher", "TEACHER", "Students"]
    + [f"S{i:02d}" for i in range(1, 41)]
)


def extract_shortcodes(codebook) -> list[str]:
    """Liste der gültigen Shortcodes aus dem Codebuch — in Reihenfolge, dedupliziert.

    Reihenfolge bleibt erhalten (für menschliche Lesbarkeit der Schemas im
    Debug-Output); Duplikate werden entfernt. Leere Codes werden gefiltert.
    """
    if not isinstance(codebook, list):
        return []
    seen = set()
    out = []
    for entry in codebook:
        code = _extract_code(entry) if isinstance(entry, dict) else ""
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def extract_speakers(transcript, lang: str = "de") -> list[str]:
    """Liste der im Transkript vorkommenden Sprecher-Labels.

    Wir nehmen den literalen String VOR dem ersten Doppelpunkt am Zeilenanfang.
    Reihenfolge: First-Seen, dedupliziert. Falls das Transkript leer/unparsbar
    ist, fallen wir auf eine Standardliste zurück, damit das enum nicht leer
    wird (ein leeres enum macht jedes JSON-Schema unsatisfiable).
    """
    text = str(transcript or "")
    seen = set()
    out = []
    for m in _SPEAKER_LINE_RE.finditer(text):
        label = m.group(1).strip()
        if not label or label in seen:
            continue
        # Sehr lange "Pseudo-Sprecher" (z.B. ganze Sätze mit Doppelpunkt)
        # rausfiltern — alles über 40 Zeichen ist mit hoher Wahrscheinlichkeit
        # kein Speaker-Label, sondern Inhalt.
        if len(label) > 40:
            continue
        seen.add(label)
        out.append(label)
    if not out:
        return list(_FALLBACK_SPEAKERS_DE if lang == "de" else _FALLBACK_SPEAKERS_EN)
    return out


def build_analysis_schema(
    codebook,
    transcript,
    *,
    lang: str = "de",
    constrain_speakers: bool = True,
) -> dict:
    """Baue ein JSON-Schema mit enum-Constraints für Shortcode (+ optional Sprecher).

    Wenn das Codebuch keine extrahierbaren Shortcodes liefert, fällt das
    Schema auf den unconstrained "string"-Typ zurück — besser als gar kein
    Schema, und die nachgelagerten Provider entscheiden, ob sie das Schema
    überhaupt nutzen.
    """
    shortcodes = extract_shortcodes(codebook)
    speakers = extract_speakers(transcript, lang=lang) if constrain_speakers else []

    shortcode_field: dict = {"type": "string", "description": "Der Shortcode aus dem Codebuch."}
    if shortcodes:
        shortcode_field["enum"] = shortcodes

    sprecher_field: dict = {"type": "string", "description": "Sprecher-Label aus dem Transkript (z.B. 'Lehrperson', 'S01')."}
    if speakers:
        sprecher_field["enum"] = speakers

    return {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "array",
                "description": "Eine Codierung pro codierbarer Äußerung. Multi-Coding: ein Item pro Code.",
                "items": {
                    "type": "object",
                    "properties": {
                        "#": {"type": "integer", "description": "Sequentielle Nummer ab 1."},
                        "Sprecher": sprecher_field,
                        "Shortcode": shortcode_field,
                        "Impuls": {"type": "string", "description": "Wörtliche Äußerung."},
                    },
                    "required": ["#", "Sprecher", "Shortcode", "Impuls"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["analysis"],
        "additionalProperties": False,
    }


def has_enum_constraints(schema: dict) -> bool:
    """True, wenn das Schema mindestens auf Shortcode ein enum trägt.

    Die Provider nutzen das, um zu loggen, ob sie tatsächlich strikter codieren
    oder de facto auf den alten Pfad zurückgefallen sind.
    """
    try:
        items = schema["properties"]["analysis"]["items"]["properties"]
        return bool(items.get("Shortcode", {}).get("enum"))
    except (KeyError, TypeError):
        return False
