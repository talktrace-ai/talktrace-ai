"""Transkript-Struktur-Analyse und konfigurierbare Konvertierung in das
Zielformat `S0X: text` / `<Lehrperson>: text` (eine Zeile pro Turn).

Dieses Modul ist UI-frei und enthält reine Logik, damit es ohne Shiny-
Kontext getestet werden kann.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# --- Regex-Konstanten -------------------------------------------------

# noScribe-Header auf eigener Zeile, z.B. "SPEAKER_00:" / "SPEAKER 1"
_SPEAKER_HEADER_RE = re.compile(
    r'^\s*"?\s*SPEAKER[_\s-]*(\d+)\s*:?\s*"?\s*$', re.IGNORECASE
)

# Inline-Sprecher am Zeilenanfang: "Frau Müller: ...", "L1: ...", "S03: ..."
# Label = Buchstabe/Umlaut, dann bis zu 30 Zeichen aus Buchstaben/Ziffern/.- /Leerzeichen.
_INLINE_SPEAKER_RE = re.compile(
    r'^\s*"?\s*([A-Za-zÄÖÜäöüß][\wÄÖÜäöüß.\- ]{0,29}?)\s*:\s+(.*)$'
)

# Bereits-valides Format `S0X:` (genutzt für Vorschau-Vergleich)
_VALID_LINE_RE = re.compile(r'^\s*"?\s*S\d{2}\s*:')

# Zeitmarken (immer entfernen). Liberalere Erkennung als zuvor:
# - Sub-Segmente mit `:` ODER `.` als Separator
# - 1..3 Sub-Segmente nach erstem Wert (MM:SS, HH:MM:SS, HH:MM:SS.mmm)
# - Range-Pfeil mit beliebigem Whitespace
_TS_NUM = r'\d{1,2}(?:[:.]\d{1,3}){1,3}'
_TIMESTAMP_PATTERNS = [
    # Range: 00:00:01.123 --> 00:00:02.456, 01:23 --> 01:45, 1.23 --> 1.45
    re.compile(rf'{_TS_NUM}\s*-->\s*{_TS_NUM}'),
    # Bracketed: [hh:mm], [hh:mm:ss], [hh:mm:ss.fff]
    re.compile(rf'\[{_TS_NUM}\]'),
    re.compile(rf'\({_TS_NUM}\)'),
    re.compile(rf'\{{{_TS_NUM}\}}'),
    # Zeilenanfang: "00:01:23 ..." / "01:23 ..." (MULTILINE)
    re.compile(rf'^\s*{_TS_NUM}\s+', re.MULTILINE),
]

# Bracket-Muster für Aktionen / Annotationen. Reihenfolge wichtig: längere
# Multichar-Delimiter vor Singlechar prüfen, damit `//...//` nicht von einem
# generischen Pattern zerlegt wird.
# Tupel: (delimiter-key, regex)
_BRACKET_PATTERNS = [
    ("[]", re.compile(r'\[[^\]\n]+\]')),
    ("()", re.compile(r'\([^)\n]+\)')),
    ("{}", re.compile(r'\{[^}\n]+\}')),
    ("<>", re.compile(r'<[^>\n]+>')),
    ("//", re.compile(r'//[^/\n]+//')),
    ("**", re.compile(r'\*[^*\n]+\*')),
]

# Sonstige verdächtige Marker (nicht-paarig). Nach Timestamp- und Bracket-
# Strip übrig gebliebene Tokens, die typischerweise aus Transkriptions-
# Tools stammen und im Zielformat stören. Reihenfolge: längere Pattern
# zuerst, damit `-->` nicht von `--`-Pattern verschluckt wird.
_OTHER_TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("-->", re.compile(r'-->')),
    ("<--", re.compile(r'<--')),
    ("==>", re.compile(r'==>')),
    ("<==", re.compile(r'<==')),
    ("//", re.compile(r'(?<!/)//(?!/)')),  # standalone //
    ("\\\\", re.compile(r'\\\\')),
    ("===", re.compile(r'={3,}')),
    ("~~~", re.compile(r'~{2,}')),
    ("***", re.compile(r'\*{3,}')),
    ("###", re.compile(r'#{2,}')),
    ("....", re.compile(r'\.{4,}')),
    ("--", re.compile(r'(?<![-<])--(?![->])')),  # doppelter Bindestrich; nicht Teil von --> oder <--
]

# Inhalte, die nach Strippen als "leer" gelten (bspw. nur Whitespace
# oder verbleibende Doppelpunkte)
_EMPTY_BODY_RE = re.compile(r'^\s*[:\-\.]*\s*$')


# --- Datenklassen ------------------------------------------------------

@dataclass
class BracketGroup:
    delimiter: str           # '[]', '()', '{}', '<>', '//', '**'
    samples: list[str] = field(default_factory=list)
    count: int = 0


@dataclass
class TokenGroup:
    token: str               # literaler Marker, z.B. '-->', '//', '==='
    count: int = 0


@dataclass
class TranscriptAnalysis:
    speakers: list[str] = field(default_factory=list)
    timestamp_patterns: list[str] = field(default_factory=list)
    bracket_patterns: list[BracketGroup] = field(default_factory=list)
    other_tokens: list[TokenGroup] = field(default_factory=list)
    speaker_format: str = "unknown"   # 'noScribe' | 'inline' | 'unknown'
    line_count: int = 0


@dataclass
class ConversionOptions:
    # raw-speaker-id -> 'TEACHER' | 'S01'..'SNN' | None (=ignorieren)
    speaker_map: dict[str, Optional[str]] = field(default_factory=dict)
    # delimiter -> True wenn entfernen
    strip_brackets: dict[str, bool] = field(default_factory=dict)
    # token-string -> True wenn entfernen
    strip_tokens: dict[str, bool] = field(default_factory=dict)
    teacher_label: str = "TEACHER"


# --- Helper -----------------------------------------------------------

def _strip_timestamps(text: str) -> tuple[str, list[str]]:
    """Entfernt alle Zeitmarken aus `text`. Gibt den bereinigten Text und
    eine Liste mit Beispielen der gefundenen Muster zurück."""
    samples: list[str] = []
    cleaned = text
    for pat in _TIMESTAMP_PATTERNS:
        found = pat.findall(cleaned)
        if found:
            samples.append(str(found[0]).strip())
        cleaned = pat.sub(" ", cleaned)
    # doppelte Whitespaces zusammenfassen, ohne Zeilenumbrüche zu zerstören
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned, samples


def _detect_brackets(text: str) -> list[BracketGroup]:
    """Findet bracket-basierte Aktionsmuster im Text (nach Zeitmarken-Strip).
    Reine Zahlen/Doppelpunkt-Inhalte werden ignoriert."""
    groups: list[BracketGroup] = []
    for delim, pat in _BRACKET_PATTERNS:
        matches = pat.findall(text)
        # Filter: keine reinen Zahl-/Zeit-artigen Reste
        filtered = [
            m for m in matches
            if not re.fullmatch(r'[\[\(\{\<\*/\s\d:.,\-\]\)\}\>]+', m)
        ]
        if not filtered:
            continue
        # Eindeutige Beispiele behalten (Reihenfolge stabil)
        seen: list[str] = []
        for m in filtered:
            if m not in seen:
                seen.append(m)
            if len(seen) >= 3:
                break
        groups.append(BracketGroup(
            delimiter=delim,
            samples=seen,
            count=len(filtered),
        ))
    return groups


def _strip_brackets_all(text: str) -> str:
    """Entfernt alle bekannten Bracket-Gruppen — nur für Detection-Pipeline."""
    out = text
    for _delim, pat in _BRACKET_PATTERNS:
        out = pat.sub(" ", out)
    return out


def _detect_other_tokens(text: str) -> list[TokenGroup]:
    """Findet sonstige verdächtige Marker (nicht-paarig). `text` sollte
    bereits ohne Zeitmarken und Brackets sein, damit Marker aus diesen
    nicht hier nochmal auftauchen."""
    groups: list[TokenGroup] = []
    seen: set[str] = set()
    for token, pat in _OTHER_TOKEN_PATTERNS:
        matches = pat.findall(text)
        if not matches:
            continue
        if token in seen:
            continue
        seen.add(token)
        groups.append(TokenGroup(token=token, count=len(matches)))
    return groups


def _detect_speakers(text: str) -> tuple[list[str], str]:
    """Erkennt eindeutige Sprecher in Reihenfolge des ersten Auftretens.
    Gibt (speakers, format) zurück; format ∈ {'noScribe', 'inline', 'unknown'}.
    """
    speakers: list[str] = []
    seen: set[str] = set()
    fmt = "unknown"

    for raw in text.splitlines():
        m = _SPEAKER_HEADER_RE.match(raw)
        if m:
            label = f"SPEAKER_{int(m.group(1)):02d}"
            if label not in seen:
                seen.add(label)
                speakers.append(label)
            if fmt == "unknown":
                fmt = "noScribe"
            continue
        m2 = _INLINE_SPEAKER_RE.match(raw)
        if m2:
            label = m2.group(1).strip()
            if label and label not in seen:
                seen.add(label)
                speakers.append(label)
            if fmt == "unknown":
                fmt = "inline"
    return speakers, fmt


# --- Public API -------------------------------------------------------

def analyze_transcript(text: str, teacher_hint: Optional[str] = None) -> TranscriptAnalysis:
    """Analysiert die Struktur eines Transkripts."""
    if not text:
        return TranscriptAnalysis()

    cleaned, ts_samples = _strip_timestamps(text)
    speakers, fmt = _detect_speakers(cleaned)
    brackets = _detect_brackets(cleaned)
    cleaned_no_brackets = _strip_brackets_all(cleaned)
    other_tokens = _detect_other_tokens(cleaned_no_brackets)

    return TranscriptAnalysis(
        speakers=speakers,
        timestamp_patterns=ts_samples,
        bracket_patterns=brackets,
        other_tokens=other_tokens,
        speaker_format=fmt,
        line_count=len(text.splitlines()),
    )


def _normalize_label(s: str) -> str:
    return re.sub(r'[^a-zA-Zäöüß]', '', s).lower()


def suggest_default_options(
    analysis: TranscriptAnalysis,
    teacher_hint: Optional[str] = None,
) -> ConversionOptions:
    """Erzeugt Default-Mapping basierend auf Heuristik."""
    teacher_label = teacher_hint.strip() if teacher_hint else "TEACHER"
    teacher_norm = _normalize_label(teacher_label) if teacher_label else ""

    speaker_map: dict[str, Optional[str]] = {}
    teacher_assigned: Optional[str] = None

    # Schritt 1: Lehrperson identifizieren
    candidates: list[str] = []
    for raw in analysis.speakers:
        norm = _normalize_label(raw)
        # Match auf hint
        if teacher_norm and (norm == teacher_norm or teacher_norm in norm or norm in teacher_norm):
            candidates.append(raw)
            continue
        # Heuristik: beginnt mit L/Lehr/T/Teach
        if re.match(r'^(lehr|teach|l\d|t\d)', norm) and not norm.startswith("s"):
            candidates.append(raw)

    if candidates:
        teacher_assigned = candidates[0]
        speaker_map[teacher_assigned] = "TEACHER"

    # Schritt 2: Restliche Sprecher zu S01..SN nummerieren
    student_idx = 1
    for raw in analysis.speakers:
        if raw == teacher_assigned:
            continue
        # Falls roher Speaker bereits S## passt und nicht als Lehrer markiert,
        # versuche numerische Reihenfolge zu erhalten — sonst neu nummerieren.
        speaker_map[raw] = f"S{student_idx:02d}"
        student_idx += 1

    # Schritt 3: Bracket-Defaults
    # `()` häufig sprachlich ("(Pause)"), Rest ist meistens Markup.
    bracket_defaults = {"[]": True, "()": False, "{}": True, "<>": True, "//": True, "**": True}
    strip_brackets: dict[str, bool] = {}
    for grp in analysis.bracket_patterns:
        strip_brackets[grp.delimiter] = bracket_defaults.get(grp.delimiter, True)

    # Schritt 4: Sonstige Tokens — alle defaultmäßig entfernen
    strip_tokens: dict[str, bool] = {grp.token: True for grp in analysis.other_tokens}

    return ConversionOptions(
        speaker_map=speaker_map,
        strip_brackets=strip_brackets,
        strip_tokens=strip_tokens,
        teacher_label=teacher_label or "TEACHER",
    )


def _strip_brackets_from_line(line: str, strip_brackets: dict[str, bool]) -> str:
    out = line
    for delim, pat in _BRACKET_PATTERNS:
        if strip_brackets.get(delim):
            out = pat.sub(" ", out)
    return out


def _strip_tokens_from_line(line: str, strip_tokens: dict[str, bool]) -> str:
    out = line
    for token, pat in _OTHER_TOKEN_PATTERNS:
        if strip_tokens.get(token):
            out = pat.sub(" ", out)
    return out


def convert_with_options(text: str, options: ConversionOptions) -> str:
    """Konvertiert `text` ins Zielformat unter Beachtung der `options`.

    Pipeline:
    1. Zeitmarken strippen
    2. Brackets pro Delimiter strippen wenn aktiviert
    3. Sprecher-Header parsen (noScribe oder inline), Body sammeln
    4. Whitespace normalisieren
    5. Mapping anwenden (None = ignorieren)
    6. Eine Zeile pro Turn ausgeben
    """
    if not text:
        return ""

    cleaned, _ = _strip_timestamps(text)

    turns: list[tuple[str, str]] = []  # (raw_speaker, body)
    current_raw: Optional[str] = None
    current_buf: list[str] = []

    def flush() -> None:
        nonlocal current_raw, current_buf
        if current_raw is None:
            current_buf = []
            return
        body = " ".join(s.strip() for s in current_buf if s.strip())
        body = _strip_brackets_from_line(body, options.strip_brackets)
        body = _strip_tokens_from_line(body, options.strip_tokens)
        body = re.sub(r"\s+", " ", body).strip()
        if body and not _EMPTY_BODY_RE.match(body):
            turns.append((current_raw, body))
        current_buf = []

    for raw in cleaned.splitlines():
        m = _SPEAKER_HEADER_RE.match(raw)
        if m:
            flush()
            current_raw = f"SPEAKER_{int(m.group(1)):02d}"
            continue
        m2 = _INLINE_SPEAKER_RE.match(raw)
        if m2:
            flush()
            current_raw = m2.group(1).strip()
            current_buf.append(m2.group(2))
            continue
        if current_raw is None:
            continue
        current_buf.append(raw)
    flush()

    out_lines: list[str] = []
    for raw_spk, body in turns:
        target = options.speaker_map.get(raw_spk)
        if target is None:
            continue
        if target == "TEACHER":
            label = options.teacher_label or "TEACHER"
        else:
            label = target
        out_lines.append(f"{label}: {body}")

    return "\n".join(out_lines)
