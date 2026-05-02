"""Auto-generated methods paragraph for papers.

The reproducibility fingerprint pins down configuration; the methods
section is the prose that surrounds it. Reviewers want to see the tool,
the model, the sample scope, and a hook to verify the run — this module
assembles all four into a paragraph the researcher can copy straight
into a manuscript without losing momentum hunting down details.

Two languages, identical structure: a single sentence on tool + model +
codebook + prompt status, a sentence on sample scope, and a sentence
that anchors reproducibility via fingerprint + date.
"""
from __future__ import annotations

from datetime import date as _date
from typing import Optional


_DEFAULT_DATE = lambda: _date.today().isoformat()


def _coerce_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _count_codes(codebook) -> int:
    """Count distinct, non-empty codes in the codebook payload.

    The app stores codebooks as a list of dicts (first column = code).
    Strings or unexpected shapes degrade to 0 so the methods text still
    renders something sensible.
    """
    if not codebook:
        return 0
    if isinstance(codebook, list):
        seen = set()
        for row in codebook:
            if isinstance(row, dict) and row:
                first_key = next(iter(row))
                code = str(row.get(first_key, "")).strip()
                if code:
                    seen.add(code)
        return len(seen)
    return 0


def build_methods_text(
    *,
    lang: str = "de",
    model: str = "",
    codebook=None,
    num_codes: Optional[int] = None,
    num_pupils=None,
    num_participants=None,
    num_impulses=None,
    num_coded=None,
    fingerprint: str = "",
    prompts_customised: bool = False,
    date_str: Optional[str] = None,
) -> str:
    """Return a 2-3 sentence methods paragraph in the given language.

    All numeric inputs are coerced defensively — the methods text is
    generated lazily from app state that may be partially populated
    (e.g. when the user opens the Results tab right after analysis).
    """
    if num_codes is None:
        num_codes = _count_codes(codebook)
    n_codes = _coerce_int(num_codes)
    n_pupils = _coerce_int(num_pupils)
    n_part = _coerce_int(num_participants)
    n_imp = _coerce_int(num_impulses)
    n_cod = _coerce_int(num_coded)
    fp = (fingerprint or "").strip() or "—"
    mdl = (model or "").strip() or ("unbekannt" if lang == "de" else "unknown")
    date_str = date_str or _DEFAULT_DATE()

    if lang == "de":
        prompt_clause = (
            "die System- und User-Prompts wurden gegenüber den Standardvorgaben angepasst"
            if prompts_customised
            else "es wurden die Standard-Prompts der Anwendung verwendet"
        )
        return (
            f"Die Codierung wurde mit TalkTrace AI neo (Filler 2026) durchgeführt. "
            f"Als Sprachmodell kam {mdl} zum Einsatz; das Codebuch umfasste {n_codes} Codes; "
            f"{prompt_clause}. "
            f"Insgesamt wurden {n_imp} Beiträge analysiert, davon {n_cod} mit einem Code versehen "
            f"({n_pupils} Schüler:innen, davon {n_part} aktiv beteiligt). "
            f"Die exakte Konfiguration (Codebuch, System- und User-Prompt, Modell, Transkript) "
            f"ist über den Reproduzierbarkeits-Hash {fp} dokumentiert; die Analyse erfolgte am {date_str}."
        )

    prompt_clause = (
        "the system and user prompts were customised from the application defaults"
        if prompts_customised
        else "the application's default prompts were used"
    )
    return (
        f"Coding was performed with TalkTrace AI neo (Filler 2026). "
        f"The language model used was {mdl}; the codebook contained {n_codes} codes; "
        f"{prompt_clause}. "
        f"A total of {n_imp} turns were analysed, of which {n_cod} received a code "
        f"({n_pupils} students, {n_part} of whom actively participated). "
        f"The exact configuration (codebook, system and user prompts, model, transcript) "
        f"is documented by the reproducibility hash {fp}; the analysis was conducted on {date_str}."
    )
