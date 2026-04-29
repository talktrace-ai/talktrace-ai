"""talktrace_ai.utils.credentials"""
import sys
import keyring
import keyring.errors

_KEYRING_WARNED = False


def _keyring_unavailable():
    global _KEYRING_WARNED
    if not _KEYRING_WARNED:
        _KEYRING_WARNED = True
        print("[TalkTrace] No system keyring available — API keys will not "
              "persist between sessions.", file=sys.stderr)


def safe_get_password(service, key):
    try:
        return keyring.get_password(service, key)
    except keyring.errors.NoKeyringError:
        _keyring_unavailable()
        return None
    except keyring.errors.KeyringError:
        return None
    except Exception:
        return None


def safe_set_password(service, key, value):
    try:
        keyring.set_password(service, key, value)
        return True
    except keyring.errors.NoKeyringError:
        _keyring_unavailable()
        return False
    except keyring.errors.KeyringError:
        return False
    except Exception:
        return False


def safe_delete_password(service, key):
    try:
        keyring.delete_password(service, key)
        return True
    except (keyring.errors.PasswordDeleteError,
            keyring.errors.NoKeyringError,
            keyring.errors.KeyringError):
        return False
    except Exception:
        return False


def keyring_available():
    try:
        backend = keyring.get_keyring()
    except Exception:
        return False
    name = (getattr(backend, "name", "") or backend.__class__.__name__).lower()
    # The "fail" backend is keyring's null backend used when no real backend
    # could be loaded; treat it as unavailable so the UI can warn the user.
    return "fail" not in name and "null" not in name


# Response-Cache: bei identischem (provider, model, system, user, transcript,
# codebook) liefern wir direkt die gespeicherte Antwort zurück. Spart komplette
# API-Calls z.B. beim Re-Run nach UI-Wechseln.
