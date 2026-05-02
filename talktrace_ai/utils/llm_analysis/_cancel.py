"""Cooperative cancellation token for LLM streaming.

A `CancelToken` wraps a `threading.Event`. The UI sets it from the main
thread; provider stream generators check it between chunks and break out
cleanly. Cancellation is best-effort and only effective for streaming —
non-streaming calls block on a single HTTP request and cannot be aborted
mid-flight without provider-SDK support we don't currently have.

Usage in a provider stream:

    for chunk in stream:
        if token.is_cancelled():
            return  # caller's finally block flushes whatever was emitted
        ...

Usage in the consumer (UI):

    token.cancel()  # set the flag

The token is reset (`token.reset()`) at the start of every analysis so a
previous cancel doesn't leak into the next run.
"""
import threading


class CancelToken:
    __slots__ = ("_event",)

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()
