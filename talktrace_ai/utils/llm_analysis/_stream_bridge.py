"""Sync-to-async bridge for provider streaming generators.

Provider SDKs (anthropic, openai, groq, ollama) are synchronous: their
streaming iterators block on network I/O. The Shiny event loop is async
and must not be blocked. We run the sync generator in a worker thread
and forward each event into an asyncio.Queue, which the handler awaits.

Each provider's streaming function is a sync generator yielding events:
    {"type": "item", "data": {<coded item>}}
    {"type": "done", "raw_json": "<full JSON for cache>", "stop_reason": "..."}
    {"type": "error", "message": "..."}

`async_stream(producer, *args, **kwargs)` converts that sync generator
into an async iterator suitable for `async for ... in ...`.
"""
import asyncio
import threading
import traceback


_SENTINEL = object()


async def async_stream(producer_fn, *args, **kwargs):
    """Run a sync generator in a worker thread and yield its events asynchronously.

    Any exception raised inside the producer is caught and surfaced as an
    {"type": "error"} event so callers see a uniform stream of dicts. The
    worker thread is daemonic — it does not block process shutdown.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def thread_target():
        try:
            for event in producer_fn(*args, **kwargs):
                loop.call_soon_threadsafe(queue.put_nowait, event)
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
