"""Backwards-compatible re-export of TRANSLATIONS.

Each language's strings now live in a sibling module (en.py, de.py); this
module just stitches them back together so existing
 callers keep working.
"""
from .en import STRINGS as _en
from .de import STRINGS as _de

TRANSLATIONS = {
    "en": _en,
    "de": _de,
}
