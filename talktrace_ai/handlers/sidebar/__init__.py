"""Sidebar section: model picker, prompts, cost, analysis, report, session.

Split into themed submodules; ``register(state)`` orchestrates them so the
public API matches the previous monolithic ``sidebar.py`` exactly.
"""
from . import _model_select, _prompts, _cost, _analysis, _report, _session


def register(state):
    _model_select.register(state)
    _prompts.register(state)        # exposes effective_*_prompt + _speaker_flags on state
    _cost.register(state)           # reads effective_*_prompt
    _session.register(state)        # exposes history_version on state
    _analysis.register(state)       # sets state.run_analysis (auto-saves bump history_version)
    _report.register(state)
