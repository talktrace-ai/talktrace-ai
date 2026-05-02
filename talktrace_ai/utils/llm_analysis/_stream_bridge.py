"""Sync-to-async bridge for provider streaming generators.

Provider SDKs (anthropic, openai, groq, ollama) are synchronous: their
streaming iterators block on network I/O. The Shiny event loop is async
and must not be blocked. We run the sync generator in a worker thread
and forward each event into an asyncio.Queue, which the handler awaits.

Each provider's streaming function is a sync generator yielding events:
    {"type": "item", "data": {<coded item>}}
    {"type": "done", "raw_json": "<full JSON for cache>", "stop_reason": "..."}
    {"type": "error", "message": "..."}
    {"type": "cancelled", "items_so_far": N}   # emitted when CancelToken is set

`async_stream(producer, *args, **kwargs)` converts that sync generator
into an async iterator suitable for `async for ... in ...`. The optional
`cancel_token` argument is bound into kwargs as `_cancel_token` so the
producer can check it between chunks. Producers that ignore the kwarg
work unchanged (kwargs.pop with a default).
"""
import asyncio
import threading
import traceback


_SENTINEL = object()


async def async_stream(producer_fn, *args, cancel_token=None, **kwargs):
    """Run a sync generator in a worker thread and yield its events asynchronously.

    Any exception raised inside the producer is caught and surfaced as an
    {"type": "error"} event so callers see a uniform stream of dicts. The
    worker thread is daemonic — it does not block process shutdown.

    If `cancel_token` is provided, it is forwarded to the producer via the
    `_cancel_token` keyword. The bridge also stops yielding to the consumer
    once the token fires, so the UI sees no further items even if the
    producer is mid-chunk when the cancel arrives.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    if cancel_token is not None:
        kwargs["_cancel_token"] = cancel_token

    def thread_target():
        try:
            for event in producer_fn(*args, **kwargs):
                loop.call_soon_threadsafe(queue.put_nowait, event)
                if cancel_token is not None and cancel_token.is_cancelled():
                    # Producer is expected to break out on its own check, but
                    # we also stop forwarding here so a slow producer can't
                    # keep updating the UI after the user clicked cancel.
                    break
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[STREAM BRIDGE] producer raised: {exc}\n{tb}")
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "error", "message": f"Streaming failed: {exc}"},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    threading.Thread(target=thread_target, daemon=True).start()

    while True:
        event = await queue.get()
        if event is _SENTINEL:
            return
        yield event
