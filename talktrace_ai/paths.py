import sys
from pathlib import Path

_WELCOME_FLAG_FILE = Path(__file__).parent / "config" / ".welcome_shown"
_DATAPROTECTION_FLAG_FILE = Path(__file__).parent / "config" / ".dataprotection_acknowledged"


def _welcome_shown():
    return _WELCOME_FLAG_FILE.exists()


def _mark_welcome_shown():
    try:
        _WELCOME_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _WELCOME_FLAG_FILE.touch()
    except OSError:
        pass


def _dataprotection_acknowledged():
    return _DATAPROTECTION_FLAG_FILE.exists()


def _mark_dataprotection_acknowledged():
    try:
        _DATAPROTECTION_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DATAPROTECTION_FLAG_FILE.touch()
    except OSError:
        pass


def resource_path(relative_path: str) -> Path:
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).parent / relative_path
