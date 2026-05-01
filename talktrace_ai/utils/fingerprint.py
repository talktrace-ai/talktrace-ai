"""Reproducibility fingerprint — short hash that pins down a coding run.

Combines the four ingredients that determine the LLM's output:
codebook, system prompt, user prompt, model name, and the transcript
content. Two runs that share the same fingerprint were produced from the
same configuration; anyone reproducing the analysis can verify alignment
at a glance without comparing hundreds of lines of prompts and codebooks.

We embed it in the legend section of every report and surface it in the
Results tab so it can be quoted in methods sections and reviewer replies.
"""
from __future__ import annotations

import hashlib


def _norm(s) -> str:
    if s is None:
        return ""
    return str(s).strip().replace("\r\n", "\n").replace("\r", "\n")


def compute_fingerprint(codebook, system_prompt, user_prompt, model, transcript) -> str:
    """Return a 12-char hex fingerprint of the four pinning ingredients.

    12 hex chars (~48 bits) is enough to make accidental collisions
    vanishingly unlikely while staying short enough to print in a report
    legend or paste into a Slack message.
    """
    h = hashlib.sha256()
    for part in (codebook, system_prompt, user_prompt, model, transcript):
        h.update(_norm(part).encode("utf-8"))
        h.update(b"\x1f")  # unit separator so concatenation can't collide
    return h.hexdigest()[:12]
