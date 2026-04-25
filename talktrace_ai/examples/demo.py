"""Synthetic demo data so new users can preview the app without API keys."""

import pandas as pd


DEMO_TEACHER_NAME = "LEHRER"
DEMO_GROUP_ID = "Demo-Klasse"
DEMO_NUM_PUPILS = 22


DEMO_TRANSCRIPT = """LEHRER: Heute geht es um den Wasserkreislauf. Wer kann sagen, wo das Wasser anfängt, wenn es regnet?
S01: Aus den Wolken kommt es runter.
LEHRER: Genau, aus den Wolken. Aber wie kommt das Wasser überhaupt in die Wolken?
S02: Es verdunstet aus dem Meer.
S03: Und aus Seen und Flüssen.
LEHRER: Sehr gut. Was ist der Unterschied zwischen Verdunstung und Sieden?
S01: Verdunstung passiert die ganze Zeit, auch ohne Hitze. Sieden ist nur bei hundert Grad.
LEHRER: Stimmt. Und wie nennt man den Vorgang, wenn aus Wasserdampf wieder flüssiges Wasser wird?
S04: Kondensation, glaube ich.
LEHRER: Richtig. Was passiert in der Wolke, dass es zu regnen anfängt?
S02: Die Tröpfchen werden so groß, dass sie zu schwer werden.
LEHRER: Genau, und dann fällt das Wasser als Regen, Schnee oder Hagel zu Boden. Wer kann mir noch ein anderes Beispiel für Niederschlag nennen?
S03: Tau am Morgen.
LEHRER: Sehr gute Beobachtung. Warum ist der Wasserkreislauf eigentlich wichtig für uns?
S01: Ohne ihn hätten wir kein Trinkwasser.
S04: Und keine Pflanzen.
LEHRER: Genau. Lasst uns das jetzt zusammen aufzeichnen.
"""


DEMO_CODE_LEGEND = "Q1=Faktenfrage, Q2=Erklärungsfrage, Q3=Vergleichsfrage, F1=Feedback bestätigend"


def build_demo_llm_analysis_df():
    """Build the synthetic LLM analysis DataFrame for the demo transcript.

    Columns match what run_analysis() produces.
    """
    rows = [
        {"#": 1, "Sprecher": "LEHRER", "Shortcode": "Q1",
         "Impuls": "Wer kann sagen, wo das Wasser anfängt, wenn es regnet?"},
        {"#": 2, "Sprecher": "LEHRER", "Shortcode": "Q2",
         "Impuls": "Aber wie kommt das Wasser überhaupt in die Wolken?"},
        {"#": 3, "Sprecher": "LEHRER", "Shortcode": "Q3",
         "Impuls": "Was ist der Unterschied zwischen Verdunstung und Sieden?"},
        {"#": 4, "Sprecher": "LEHRER", "Shortcode": "Q1",
         "Impuls": "Wie nennt man den Vorgang, wenn aus Wasserdampf wieder flüssiges Wasser wird?"},
        {"#": 5, "Sprecher": "LEHRER", "Shortcode": "Q2",
         "Impuls": "Was passiert in der Wolke, dass es zu regnen anfängt?"},
        {"#": 6, "Sprecher": "LEHRER", "Shortcode": "F1",
         "Impuls": "Genau, und dann fällt das Wasser als Regen, Schnee oder Hagel zu Boden."},
        {"#": 7, "Sprecher": "LEHRER", "Shortcode": "Q1",
         "Impuls": "Wer kann mir noch ein anderes Beispiel für Niederschlag nennen?"},
        {"#": 8, "Sprecher": "LEHRER", "Shortcode": "Q2",
         "Impuls": "Warum ist der Wasserkreislauf eigentlich wichtig für uns?"},
    ]
    return pd.DataFrame(rows, columns=["#", "Sprecher", "Shortcode", "Impuls"])
