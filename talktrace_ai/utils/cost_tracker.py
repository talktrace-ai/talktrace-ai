"""Cumulative cost tracker — persisted across sessions.

Records every successful LLM analysis as one entry in a JSON log so the
Options tab can show total spend per provider (and optionally per project).
The per-run cost prediction in the sidebar only covers the *current* run;
this module aggregates the running total.

Storage: ``talktrace_ai/config/cost_log.json`` next to ``config.ini`` so the
log survives between sessions but stays local to one install.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

_COST_LOG_FILE = Path(__file__).parent.parent / "config" / "cost_log.json"


def _empty_log():
    return {"entries": []}


def _read_log() -> dict:
    if not _COST_LOG_FILE.exists():
        return _empty_log()
    try:
        with open(_COST_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "entries" not in data:
            return _empty_log()
        return data
    except (json.JSONDecodeError, OSError):
        return _empty_log()


def _write_log(data: dict) -> bool:
    try:
        _COST_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_COST_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


def record_run(provider: str, model: str, cost_eur: float,
               input_tokens: Optional[int] = None,
               group_id: Optional[str] = None) -> bool:
    """Append one analysis run to the log. Returns True on success.

    Silently no-ops on Ollama or zero-cost runs to keep the log focused on
    actual spend (Ollama input/output cost defaults to 0.0 in the registry).
    """
    if cost_eur is None or cost_eur <= 0:
        return False
    log = _read_log()
    log["entries"].append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "provider": (provider or "").lower(),
        "model": model or "",
        "cost_eur": float(cost_eur),
        "input_tokens": int(input_tokens) if input_tokens else None,
        "group_id": group_id or "",
    })
    return _write_log(log)


def get_summary() -> dict:
    """Compute per-provider, per-model, and grand totals."""
    log = _read_log()
    by_provider: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    total_cost = 0.0
    n_runs = 0
    for entry in log.get("entries", []):
        cost = float(entry.get("cost_eur") or 0.0)
        provider = entry.get("provider") or "—"
        model = entry.get("model") or "—"
        total_cost += cost
        n_runs += 1
        by_provider.setdefault(provider, {"cost": 0.0, "runs": 0})
        by_provider[provider]["cost"] += cost
        by_provider[provider]["runs"] += 1
        by_model.setdefault(model, {"cost": 0.0, "runs": 0, "provider": provider})
        by_model[model]["cost"] += cost
        by_model[model]["runs"] += 1
    return {
        "total_cost": total_cost,
        "n_runs": n_runs,
        "by_provider": by_provider,
        "by_model": by_model,
    }


def reset_log() -> bool:
    return _write_log(_empty_log())
