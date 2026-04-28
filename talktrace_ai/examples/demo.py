"""Synthetic demo data so new users can preview the app without API keys.

The transcript, the teacher tag and the analysis rows are language-aware:
look up the matching language with `[lang]` (or pass `lang="en"`/`"de"` to
`build_demo_llm_analysis_df`). The DataFrame *column* names ("Sprecher",
"Shortcode", "Impuls") are internal keys used across the analysis pipeline
and stay German on purpose.
"""

import pandas as pd


DEMO_NUM_PUPILS = 22

DEMO_TEACHER_NAME = {
    "de": "LEHRER",
    "en": "TEACHER",
}

DEMO_GROUP_ID = {
    "de": "Demo-Klasse",
    "en": "Demo Class",
}


DEMO_TRANSCRIPT = {
    "de": """LEHRER: Wer kann sagen, wo das Wasser anfängt, wenn es regnet?
S01: Aus den Wolken kommt es runter.
LEHRER: Genau, aus den Wolken.
LEHRER: Aber wie kommt das Wasser überhaupt in die Wolken?
S02: Es verdunstet aus dem Meer.
S03: Und aus Seen und Flüssen.
LEHRER: Sehr gut.
LEHRER: Was ist der Unterschied zwischen Verdunstung und Sieden?
S01: Verdunstung passiert die ganze Zeit, auch ohne Hitze. Sieden ist nur bei hundert Grad.
LEHRER: Stimmt.
LEHRER: Wie nennt man den Vorgang, wenn aus Wasserdampf wieder flüssiges Wasser wird?
S04: Kondensation, glaube ich.
LEHRER: Richtig.
LEHRER: Was passiert in der Wolke, dass es zu regnen anfängt?
S02: Die Tröpfchen werden so groß, dass sie zu schwer werden.
LEHRER: Genau, und dann fällt das Wasser als Regen, Schnee oder Hagel zu Boden.
LEHRER: Wer kann mir noch ein anderes Beispiel für Niederschlag nennen?
S03: Tau am Morgen.
LEHRER: Sehr gute Beobachtung.
LEHRER: Warum ist der Wasserkreislauf eigentlich wichtig für uns?
S01: Ohne ihn hätten wir kein Trinkwasser.
S04: Und keine Pflanzen.
""",
    "en": """TEACHER: Who can tell me where water starts when it rains?
S01: It comes down from the clouds.
TEACHER: Exactly, from the clouds.
TEACHER: But how does the water get into the clouds in the first place?
S02: It evaporates from the sea.
S03: And from lakes and rivers.
TEACHER: Very good.
TEACHER: What is the difference between evaporation and boiling?
S01: Evaporation happens all the time, even without heat. Boiling only happens at one hundred degrees.
TEACHER: Right.
TEACHER: What do we call the process when water vapour turns back into liquid water?
S04: Condensation, I think.
TEACHER: Correct.
TEACHER: What happens inside the cloud that makes it start to rain?
S02: The droplets get so big that they become too heavy.
TEACHER: Exactly, and then the water falls to the ground as rain, snow or hail.
TEACHER: Can anyone give me another example of precipitation?
S03: Dew in the morning.
TEACHER: A very nice observation.
TEACHER: Why is the water cycle actually important for us?
S01: Without it we would have no drinking water.
S04: And no plants.
""",
}


DEMO_CODE_LEGEND = {
    "de": (
        "Q1=Faktenfrage, Q2=Erklärungsfrage, Q3=Vergleichsfrage, F1=Feedback bestätigend, "
        "A1=Antwort kurz, A2=Antwort elaboriert, A3=Beispiel/Beobachtung"
    ),
    "en": (
        "Q1=factual question, Q2=explanation question, Q3=comparison question, F1=affirmative feedback, "
        "A1=short answer, A2=elaborated answer, A3=example/observation"
    ),
}


# Strukturiertes Demo-Codebuch — list[dict] genau wie ein per docx-Tabelle
# importiertes Codebuch. Wird in der Vorschau als Tabelle gerendert
# (`show_codebook_preview` in handlers/analysis.py).
DEMO_CODEBOOK = {
    "de": [
        {"Code": "Q1", "Bezeichnung": "Faktenfrage",
         "Beschreibung": "Frage nach einem konkreten, eindeutig beantwortbaren Sachverhalt."},
        {"Code": "Q2", "Bezeichnung": "Erklärungsfrage",
         "Beschreibung": "Frage, die eine Begründung oder einen Mechanismus erfordert."},
        {"Code": "Q3", "Bezeichnung": "Vergleichsfrage",
         "Beschreibung": "Frage, die zwei oder mehr Konzepte gegenüberstellt."},
        {"Code": "F1", "Bezeichnung": "Feedback bestätigend",
         "Beschreibung": "Kurze Rückmeldung der Lehrperson, die eine Schüler-Äußerung bestätigt."},
        {"Code": "A1", "Bezeichnung": "Antwort kurz",
         "Beschreibung": "Knappe Schüler-Äußerung, ein Wort bis ein Satz."},
        {"Code": "A2", "Bezeichnung": "Antwort elaboriert",
         "Beschreibung": "Ausführlichere Schüler-Antwort mit Erklärung oder Begründung."},
        {"Code": "A3", "Bezeichnung": "Beispiel/Beobachtung",
         "Beschreibung": "Schüler-Äußerung mit konkretem Beispiel oder eigener Beobachtung."},
    ],
    "en": [
        {"Code": "Q1", "Label": "Factual question",
         "Description": "Question about a concrete, clearly answerable fact."},
        {"Code": "Q2", "Label": "Explanation question",
         "Description": "Question requiring a reason or mechanism."},
        {"Code": "Q3", "Label": "Comparison question",
         "Description": "Question contrasting two or more concepts."},
        {"Code": "F1", "Label": "Affirmative feedback",
         "Description": "Short teacher reply confirming a student utterance."},
        {"Code": "A1", "Label": "Short answer",
         "Description": "Brief student utterance, one word to one sentence."},
        {"Code": "A2", "Label": "Elaborated answer",
         "Description": "More detailed student answer with explanation or reasoning."},
        {"Code": "A3", "Label": "Example / observation",
         "Description": "Student utterance with a concrete example or own observation."},
    ],
}


_DEMO_ROWS = {
    "de": [
        ("LEHRER", "Q1", "Wer kann sagen, wo das Wasser anfängt, wenn es regnet?"),
        ("S01",    "A1", "Aus den Wolken kommt es runter."),
        ("LEHRER", "F1", "Genau, aus den Wolken."),
        ("LEHRER", "Q2", "Aber wie kommt das Wasser überhaupt in die Wolken?"),
        ("S02",    "A1", "Es verdunstet aus dem Meer."),
        ("S03",    "A1", "Und aus Seen und Flüssen."),
        ("LEHRER", "F1", "Sehr gut."),
        ("LEHRER", "Q3", "Was ist der Unterschied zwischen Verdunstung und Sieden?"),
        ("S01",    "A2", "Verdunstung passiert die ganze Zeit, auch ohne Hitze. Sieden ist nur bei hundert Grad."),
        ("LEHRER", "F1", "Stimmt."),
        ("LEHRER", "Q1", "Wie nennt man den Vorgang, wenn aus Wasserdampf wieder flüssiges Wasser wird?"),
        ("S04",    "A1", "Kondensation, glaube ich."),
        ("LEHRER", "F1", "Richtig."),
        ("LEHRER", "Q2", "Was passiert in der Wolke, dass es zu regnen anfängt?"),
        ("S02",    "A2", "Die Tröpfchen werden so groß, dass sie zu schwer werden."),
        ("LEHRER", "F1", "Genau, und dann fällt das Wasser als Regen, Schnee oder Hagel zu Boden."),
        ("LEHRER", "Q1", "Wer kann mir noch ein anderes Beispiel für Niederschlag nennen?"),
        ("S03",    "A3", "Tau am Morgen."),
        ("LEHRER", "F1", "Sehr gute Beobachtung."),
        ("LEHRER", "Q2", "Warum ist der Wasserkreislauf eigentlich wichtig für uns?"),
        ("S01",    "A1", "Ohne ihn hätten wir kein Trinkwasser."),
        ("S04",    "A1", "Und keine Pflanzen."),
    ],
    "en": [
        ("TEACHER", "Q1", "Who can tell me where water starts when it rains?"),
        ("S01",     "A1", "It comes down from the clouds."),
        ("TEACHER", "F1", "Exactly, from the clouds."),
        ("TEACHER", "Q2", "But how does the water get into the clouds in the first place?"),
        ("S02",     "A1", "It evaporates from the sea."),
        ("S03",     "A1", "And from lakes and rivers."),
        ("TEACHER", "F1", "Very good."),
        ("TEACHER", "Q3", "What is the difference between evaporation and boiling?"),
        ("S01",     "A2", "Evaporation happens all the time, even without heat. Boiling only happens at one hundred degrees."),
        ("TEACHER", "F1", "Right."),
        ("TEACHER", "Q1", "What do we call the process when water vapour turns back into liquid water?"),
        ("S04",     "A1", "Condensation, I think."),
        ("TEACHER", "F1", "Correct."),
        ("TEACHER", "Q2", "What happens inside the cloud that makes it start to rain?"),
        ("S02",     "A2", "The droplets get so big that they become too heavy."),
        ("TEACHER", "F1", "Exactly, and then the water falls to the ground as rain, snow or hail."),
        ("TEACHER", "Q1", "Can anyone give me another example of precipitation?"),
        ("S03",     "A3", "Dew in the morning."),
        ("TEACHER", "F1", "A very nice observation."),
        ("TEACHER", "Q2", "Why is the water cycle actually important for us?"),
        ("S01",     "A1", "Without it we would have no drinking water."),
        ("S04",     "A1", "And no plants."),
    ],
}


def build_demo_llm_analysis_df(lang: str = "de"):
    """Build the synthetic LLM analysis DataFrame for the demo transcript.

    Columns match what run_analysis() produces. The column *names* stay
    German on purpose because they are internal keys used across the
    analysis pipeline; only the values are language-dependent.
    """
    rows = _DEMO_ROWS.get(lang, _DEMO_ROWS["de"])
    df_rows = [
        {"#": i, "Sprecher": spk, "Shortcode": code, "Impuls": text}
        for i, (spk, code, text) in enumerate(rows, start=1)
    ]
    return pd.DataFrame(df_rows, columns=["#", "Sprecher", "Shortcode", "Impuls"])
