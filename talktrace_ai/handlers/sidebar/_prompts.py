"""LLM/speaker switches + prompt sanitization + effective prompt reactives.

Exposes ``state.effective_system_prompt``, ``state.effective_user_prompt``,
and ``state._speaker_flags`` so cost prediction and analysis can read them.
"""
from .._common import *


def register(state):
    input = state.input
    t = state.t
    system_prompt = state.system_prompt
    user_prompt = state.user_prompt

    # LLM Analyse
    @render.ui
    def loc_llm_switch():
        return ui.div(
            ui.input_switch("llm_switch", t("sidebar", "llm_switch"), True),
            class_="ttai-switch-compact",
            **{"data-tt-help": t("onboarding", "tooltip_llm_switch")},
        )

    # Sprechakt-Auswahl: nur sichtbar, wenn LLM-Analyse aktiv ist
    @render.ui
    def loc_analyse_speakers_switches():
        if not input.llm_switch():
            return None
        return ui.div(
            ui.input_switch("analyse_teacher_switch", t("sidebar", "analyse_teacher_switch"), True),
            ui.input_switch("analyse_students_switch", t("sidebar", "analyse_students_switch"), True),
            ui.input_switch("multi_coding_switch", t("sidebar", "multi_coding_switch"), False),
            class_="ttai-switch-compact",
        )

    # Effektive Prompts: Basis-Prompt + Zusatzanweisung je nach Sprecher-Auswahl.
    # Wird sowohl in der Options-Anzeige als auch beim LLM-Call verwendet,
    # damit der User sieht, was tatsächlich ans Modell geschickt wird.
    def _speaker_flags():
        # Switches werden nur gerendert, wenn llm_switch aktiv ist;
        # fallback auf True (Default), solange sie nicht existieren.
        try:
            teacher = bool(input.analyse_teacher_switch())
        except Exception:
            teacher = True
        try:
            students = bool(input.analyse_students_switch())
        except Exception:
            students = True
        return teacher, students

    def _multi_coding_flag() -> bool:
        """Schalter-Wert defensiv lesen — der Switch wird nur gerendert,
        solange ``llm_switch`` aktiv ist. Default OFF.
        """
        try:
            return bool(input.multi_coding_switch())
        except Exception:
            return False

    def _multi_coding_suffix(kind: str = "system") -> str:
        """Liefert den Prompt-Zusatz, der dem LLM mitteilt, ob Mehrfach-
        Codierung erlaubt/erwünscht (ON) oder verboten (OFF) ist. Wird
        sowohl an System- als auch an User-Prompt angehängt, damit das
        Modell unmissverständlich weiß, was es tun soll. Post-Processing
        (Hierarchie + drop_duplicates / groupby) bleibt als Sicherheitsnetz."""
        prefix = "user_prompt_multi_coding" if kind == "user" else "prompt_multi_coding"
        key = f"{prefix}_{'on' if _multi_coding_flag() else 'off'}"
        return t("sidebar", key)

    def _speaker_filter_suffix(kind: str = "system"):
        teacher, students = _speaker_flags()
        prefix = "user_prompt_filter" if kind == "user" else "prompt_filter"
        teacher_name = input.name_teacher() or t("analysis", "name_teacher_var")
        if teacher and students:
            return ""
        if teacher and not students:
            return t("sidebar", f"{prefix}_teacher_only").format(teacher_name=teacher_name)
        if students and not teacher:
            return t("sidebar", f"{prefix}_students_only").format(teacher_name=teacher_name)
        return t("sidebar", f"{prefix}_none")

    def _sanitize_prompt_for_speakers(text: str, teacher: bool, students: bool) -> str:
        """Remove teacher/student references from prompt text when that group is disabled.
        Keeps the text grammatical for the default prompts; custom prompts are
        handled best-effort."""
        if teacher and students:
            return text
        # --- German references -------------------------------------------------
        if not teacher:
            text = text.replace("Lehrperson UND Schüler:innen", "Schüler:innen")
            text = text.replace("Lehrperson und Schüler:innen", "Schüler:innen")
            text = text.replace("ALLER Sprecher:innen (Lehrperson UND Schüler:innen)", "der Schüler:innen")
            text = text.replace("ALLER Sprecher:innen", "der Schüler:innen")
            text = text.replace("Lehrperson", "")
            text = text.replace("LEHRER", "")
            text = text.replace("Lehrer", "")
        if not students:
            text = text.replace("Lehrperson UND Schüler:innen", "Lehrperson")
            text = text.replace("Lehrperson und Schüler:innen", "Lehrperson")
            text = text.replace("ALLER Sprecher:innen (Lehrperson UND Schüler:innen)", "der Lehrperson")
            text = text.replace("ALLER Sprecher:innen", "der Lehrperson")
            text = text.replace("Schüler:innen", "")
            text = text.replace("S01, S02, S03…", "")
            text = text.replace("S01, S02, S03...", "")
            text = text.replace("S01, S02, S03", "")
            text = text.replace("S01", "")
        # --- English references ------------------------------------------------
        if not teacher:
            text = text.replace("teacher AND students", "students")
            text = text.replace("teacher and students", "students")
            text = text.replace("ALL speakers (teacher AND students)", "students")
            text = text.replace("ALL speakers", "students")
            text = text.replace("teacher", "")
        if not students:
            text = text.replace("teacher AND students", "teacher")
            text = text.replace("teacher and students", "teacher")
            text = text.replace("ALL speakers (teacher AND students)", "teacher")
            text = text.replace("ALL speakers", "teacher")
            text = text.replace("students", "")
            text = text.replace("S01, S02, S03…", "")
            text = text.replace("S01, S02, S03...", "")
            text = text.replace("S01, S02, S03", "")
            text = text.replace("S01", "")
        # --- Cleanup whitespace artifacts --------------------------------------
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*-\s*", "-", text)
        return text.strip()

    @reactive.calc
    def effective_system_prompt():
        teacher, students = _speaker_flags()
        base = _sanitize_prompt_for_speakers(system_prompt.get(), teacher, students)
        return base + _speaker_filter_suffix("system") + _multi_coding_suffix("system")

    @reactive.calc
    def effective_user_prompt():
        teacher, students = _speaker_flags()
        raw = _sanitize_prompt_for_speakers(user_prompt.get(), teacher, students)
        # Beide Instruktions-Suffixe (Sprecher-Filter + Multi-Coding) werden
        # gemeinsam direkt nach dem {transcript}-Block platziert. Hintergrund:
        # LLMs leiden bei sehr langen Kontexten unter "lost in the middle" —
        # Anweisungen über Output-Format und Filter müssen nahe am Transkript
        # sitzen, nicht am Ende nach tausenden Token Codebook.
        combined = _speaker_filter_suffix("user") + _multi_coding_suffix("user")
        if not combined:
            return raw
        if "{transcript}" in raw:
            target = "{transcript}"
            idx = raw.index(target)
            insert_pos = idx + len(target)
            return raw[:insert_pos] + "\n\n" + combined + raw[insert_pos:]
        return raw + combined

    state.effective_system_prompt = effective_system_prompt
    state.effective_user_prompt = effective_user_prompt
    state._speaker_flags = _speaker_flags
