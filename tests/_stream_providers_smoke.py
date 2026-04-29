"""End-to-end smoke tests for the four provider streaming generators.

We do not call real APIs — instead we feed each provider a fake SDK client
that yields realistic event shapes (matching what the actual SDKs emit).
The tests assert that:
  - items arrive in order via {"type": "item"} events
  - each item has the four expected fields and integer numbering
  - a {"type": "done"} event closes the stream
  - the cache is populated with a parseable raw_json
  - a cache hit on a second call replays items without invoking the SDK

Run:
    PYTHONPATH=. python tests/_stream_providers_smoke.py
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import sys
sys.path.insert(0, ".")  # safety net if PYTHONPATH wasn't set

from talktrace_ai.utils import llm_cache as _cache_mod
from talktrace_ai.utils.llm_analysis import (
    llm_analysis_anthropic_stream,
    llm_analysis_openai_stream,
    llm_analysis_groq_stream,
    llm_analysis_ollama_stream,
)


def _reset_cache():
    _cache_mod._response_cache.clear()


def _drain(gen):
    """Materialise a sync generator into a list of events."""
    return list(gen)


def _assert_item_sequence(events, expected_shortcodes):
    items = [e for e in events if e["type"] == "item"]
    assert len(items) == len(expected_shortcodes), (
        f"expected {len(expected_shortcodes)} items, got {len(items)}: {items}"
    )
    for i, (ev, sc) in enumerate(zip(items, expected_shortcodes), 1):
        d = ev["data"]
        assert d["Shortcode"] == sc, f"item {i}: expected Shortcode={sc}, got {d}"
        assert d["#"] == i, f"item {i}: expected #={i}, got #={d['#']}"
        for field in ("#", "Sprecher", "Shortcode", "Impuls"):
            assert field in d, f"item {i}: missing field {field}"


def _assert_done(events):
    assert events[-1]["type"] == "done", f"last event should be done, got {events[-1]}"
    assert events[-1].get("raw_json"), "done event should carry raw_json"
    parsed = json.loads(events[-1]["raw_json"])
    assert "analysis" in parsed and isinstance(parsed["analysis"], list)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class _FakeAnthropicStreamCtx:
    """Mimics the context manager returned by client.messages.stream(...)."""

    def __init__(self, deltas, final_msg):
        self._deltas = deltas
        self._final_msg = final_msg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for chunk in self._deltas:
            yield SimpleNamespace(delta=SimpleNamespace(
                type="input_json_delta", partial_json=chunk
            ))

    def get_final_message(self):
        return self._final_msg


class _FakeAnthropicClient:
    def __init__(self, deltas, tool_input):
        self._deltas = deltas
        # Build a Message-like object with the tool_use block.
        block = SimpleNamespace(type="tool_use", name="submit_analysis", input=tool_input)
        self._final = SimpleNamespace(content=[block], stop_reason="end_turn", usage=None)

    @property
    def messages(self):
        return self

    def stream(self, **kwargs):
        return _FakeAnthropicStreamCtx(self._deltas, self._final)


def test_anthropic_stream_progressive():
    _reset_cache()
    full_obj = {"analysis": [
        {"#": 1, "Sprecher": "L", "Shortcode": "Q1", "Impuls": "Hallo"},
        {"#": 2, "Sprecher": "S01", "Shortcode": "A1", "Impuls": "Hi"},
        {"#": 3, "Sprecher": "S02", "Shortcode": "B2", "Impuls": "Sup"},
    ]}
    full_text = json.dumps(full_obj)
    # Split into 4 deltas to simulate a real stream.
    splits = [0, 30, 70, 110, len(full_text)]
    deltas = [full_text[a:b] for a, b in zip(splits[:-1], splits[1:])]

    client = _FakeAnthropicClient(deltas, full_obj)
    events = _drain(llm_analysis_anthropic_stream(
        system_prompt="sys", user_prompt="usr {transcript} {codebook}",
        model="claude-sonnet-4-5-20250929",
        transcript="t", codebook=[{"Code": "Q1"}],
        client=client,
    ))

    _assert_item_sequence(events, ["Q1", "A1", "B2"])
    _assert_done(events)
    print("OK anthropic: 3 items via input_json_delta, done with raw_json")


def test_anthropic_cache_replay():
    _reset_cache()
    full_obj = {"analysis": [
        {"#": 1, "Sprecher": "L", "Shortcode": "Q1", "Impuls": "first"},
    ]}
    deltas = [json.dumps(full_obj)]
    client = _FakeAnthropicClient(deltas, full_obj)

    # First run populates the cache.
    args = dict(
        system_prompt="sys", user_prompt="u", model="claude-haiku-4-5-20251001",
        transcript="t", codebook=[{"Code": "Q1"}], client=client,
    )
    _drain(llm_analysis_anthropic_stream(**args))

    # Second run with a client that would crash if invoked — proves cache hit.
    class _BoomClient:
        @property
        def messages(self):
            raise AssertionError("cache should have prevented SDK call")
    args["client"] = _BoomClient()
    events = _drain(llm_analysis_anthropic_stream(**args))
    _assert_item_sequence(events, ["Q1"])
    assert events[-1]["stop_reason"] == "cache_hit"
    print("OK anthropic: cache hit replays without SDK")


def test_anthropic_recovery_from_stringified_array():
    """Claude 4.x sometimes returns analysis as a JSON-encoded string under
    forced tool_use. The streaming function must recover via the existing
    helpers — same fallback as the classic path."""
    _reset_cache()
    inner = [
        {"#": 1, "Sprecher": "L", "Shortcode": "Q1", "Impuls": "x"},
        {"#": 2, "Sprecher": "S01", "Shortcode": "A1", "Impuls": "y"},
    ]
    # Tool input has analysis as a STRING, not a list.
    tool_input = {"analysis": json.dumps(inner)}
    # No useful deltas — recovery path kicks in from final_msg.
    client = _FakeAnthropicClient(deltas=[], tool_input=tool_input)
    events = _drain(llm_analysis_anthropic_stream(
        system_prompt="sys", user_prompt="u", model="claude-opus-4-6",
        transcript="t", codebook=[{"Code": "Q1"}], client=client,
    ))
    _assert_item_sequence(events, ["Q1", "A1"])
    _assert_done(events)
    print("OK anthropic: recovers when SDK delivers stringified array")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class _FakeOpenAIClient:
    def __init__(self, deltas, final_text):
        self._deltas = deltas
        self._final_text = final_text

    @property
    def responses(self):
        return self

    def create(self, **kwargs):
        # Yield delta events, then a completed event with the final text.
        for chunk in self._deltas:
            yield SimpleNamespace(type="response.output_text.delta", delta=chunk)
        yield SimpleNamespace(
            type="response.output_text.done", text=self._final_text
        )
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text=self._final_text),
        )


def test_openai_stream_progressive():
    _reset_cache()
    full_obj = {"analysis": [
        {"#": 1, "Sprecher": "L", "Shortcode": "Q1", "Impuls": "a"},
        {"#": 2, "Sprecher": "S01", "Shortcode": "A1", "Impuls": "b"},
    ]}
    full_text = json.dumps(full_obj)
    deltas = [full_text[i:i + 25] for i in range(0, len(full_text), 25)]
    client = _FakeOpenAIClient(deltas, full_text)
    events = _drain(llm_analysis_openai_stream(
        system_prompt="sys", user_prompt="usr {transcript} {codebook}",
        model="gpt-4o", transcript="t", codebook=[{"Code": "Q1"}], client=client,
    ))
    _assert_item_sequence(events, ["Q1", "A1"])
    _assert_done(events)
    print("OK openai: 2 items via output_text.delta, done with raw_json")


def test_openai_final_text_fallback():
    """If somehow no deltas surfaced item completion mid-stream, the final
    text fallback should still parse the full response."""
    _reset_cache()
    full_obj = {"analysis": [
        {"#": 1, "Sprecher": "L", "Shortcode": "Q1", "Impuls": "x"},
    ]}
    full_text = json.dumps(full_obj)
    # All in one chunk, but it arrives only as final text — simulate a
    # client that delivers the response in a single .completed event.
    class _OneShotClient:
        @property
        def responses(self): return self
        def create(self, **kwargs):
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(output_text=full_text),
            )
    events = _drain(llm_analysis_openai_stream(
        system_prompt="s", user_prompt="u", model="gpt-4o",
        transcript="t", codebook=[{"Code": "Q1"}], client=_OneShotClient(),
    ))
    _assert_item_sequence(events, ["Q1"])
    _assert_done(events)
    print("OK openai: final-text fallback when no progressive deltas")


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

def _groq_chunk(text):
    return SimpleNamespace(choices=[
        SimpleNamespace(delta=SimpleNamespace(content=text))
    ])


class _FakeGroqClient:
    def __init__(self, chunks):
        self._chunks = chunks

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        for c in self._chunks:
            yield _groq_chunk(c)


def test_groq_stream_jsonl():
    _reset_cache()
    lines = [
        '{"#": 1, "Sprecher": "L", "Shortcode": "Q1", "Impuls": "Hallo"}\n',
        '{"#": 2, "Sprecher": "S01", "Shortcode": "A1", "Impuls": "Hi"}\n',
        '{"#": 3, "Sprecher": "S02", "Shortcode": "B2", "Impuls": "Sup"}\n',
    ]
    # Split lines across multiple chunks to simulate streaming arrival.
    chunks = []
    for line in lines:
        # Split mid-line to make sure buffering works.
        chunks.append(line[:20])
        chunks.append(line[20:])
    client = _FakeGroqClient(chunks)
    events = _drain(llm_analysis_groq_stream(
        system_prompt="sys", user_prompt="usr {transcript} {codebook}",
        model="llama-3.3-70b-versatile",
        transcript="t", codebook=[{"Code": "Q1"}], client=client,
    ))
    _assert_item_sequence(events, ["Q1", "A1", "B2"])
    _assert_done(events)
    print("OK groq: 3 JSONL items across split chunks")


def test_groq_filters_fences_and_garbage():
    _reset_cache()
    chunks = [
        "```json\n",
        '{"#": 1, "Sprecher": "L", "Shortcode": "Q1", "Impuls": "x"}\n',
        "garbage line\n",
        '{"#": 2, "Sprecher": "S01", "Shortcode": "A1", "Impuls": "y"}\n',
        "```\n",
    ]
    client = _FakeGroqClient(chunks)
    events = _drain(llm_analysis_groq_stream(
        system_prompt="s", user_prompt="u", model="m",
        transcript="t", codebook=[], client=client,
    ))
    _assert_item_sequence(events, ["Q1", "A1"])
    print("OK groq: fences and unparseable lines are dropped silently")


def test_groq_no_items_yields_error():
    _reset_cache()
    client = _FakeGroqClient(["nothing useful here\n", "still nothing\n"])
    events = _drain(llm_analysis_groq_stream(
        system_prompt="s", user_prompt="u", model="m",
        transcript="t", codebook=[], client=client,
    ))
    assert any(e["type"] == "error" for e in events), (
        f"expected error event, got {events}"
    )
    print("OK groq: empty output produces error event")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _ollama_chunk(content="", thinking="", done=False, done_reason=None):
    msg = SimpleNamespace(content=content, thinking=thinking)
    chunk = SimpleNamespace(message=msg, done=done, done_reason=done_reason)
    if done:
        chunk.eval_count = 100
        chunk.eval_duration = int(1e9)  # 1s
        chunk.prompt_eval_count = 50
        chunk.prompt_eval_duration = int(5e8)
    return chunk


class _FakeOllamaClient:
    def __init__(self, chunks):
        self._chunks = chunks

    def chat(self, **kwargs):
        return iter(self._chunks)


def test_ollama_stream_jsonl():
    _reset_cache()
    text = (
        '{"#": 1, "Sprecher": "L", "Shortcode": "Q1", "Impuls": "a"}\n'
        '{"#": 2, "Sprecher": "S01", "Shortcode": "A1", "Impuls": "b"}\n'
    )
    chunks = []
    # Split into ~15-char fragments.
    for i in range(0, len(text), 15):
        chunks.append(_ollama_chunk(content=text[i:i + 15]))
    chunks.append(_ollama_chunk(done=True, done_reason="stop"))

    fake = _FakeOllamaClient(chunks)
    with patch("talktrace_ai.utils.llm_analysis.ollama.OllamaClient", return_value=fake):
        events = _drain(llm_analysis_ollama_stream(
            system_prompt="sys", user_prompt="usr {transcript} {codebook}",
            model="llama3", transcript="t", codebook=[{"Code": "Q1"}],
        ))
    _assert_item_sequence(events, ["Q1", "A1"])
    _assert_done(events)
    print("OK ollama: 2 JSONL items across fragmented chunks")


def test_ollama_done_reason_length_keeps_items():
    """When done_reason='length' we want any successfully parsed items to
    survive — truncation should not drop earlier items."""
    _reset_cache()
    text = (
        '{"#": 1, "Sprecher": "L", "Shortcode": "Q1", "Impuls": "ok"}\n'
        '{"#": 2, "Sprecher": "S01", "Shortcode": "A'  # truncated mid-line
    )
    chunks = [_ollama_chunk(content=text), _ollama_chunk(done=True, done_reason="length")]
    fake = _FakeOllamaClient(chunks)
    with patch("talktrace_ai.utils.llm_analysis.ollama.OllamaClient", return_value=fake):
        events = _drain(llm_analysis_ollama_stream(
            system_prompt="s", user_prompt="u", model="kimi-k2.6:cloud",
            transcript="t", codebook=[],
        ))
    items = [e for e in events if e["type"] == "item"]
    assert len(items) == 1, f"truncation should preserve the 1 complete item, got {len(items)}"
    assert items[0]["data"]["Shortcode"] == "Q1"
    # Done event should carry the length reason.
    assert events[-1]["type"] == "done"
    assert events[-1]["stop_reason"] == "length"
    print("OK ollama: truncation preserves complete items, surfaces done_reason=length")


# ---------------------------------------------------------------------------
# Bridge integration: async consumption with throttling shape
# ---------------------------------------------------------------------------

def test_bridge_consumes_provider_stream():
    """The async_stream bridge must forward provider events without
    reordering or losing them."""
    _reset_cache()
    import asyncio
    from talktrace_ai.utils.llm_analysis._stream_bridge import async_stream

    full_obj = {"analysis": [
        {"#": 1, "Sprecher": "L", "Shortcode": "Q1", "Impuls": "a"},
        {"#": 2, "Sprecher": "S01", "Shortcode": "A1", "Impuls": "b"},
    ]}
    text = json.dumps(full_obj)
    deltas = [text[i:i + 30] for i in range(0, len(text), 30)]
    client = _FakeAnthropicClient(deltas, full_obj)

    async def collect():
        out = []
        async for ev in async_stream(
            llm_analysis_anthropic_stream,
            "sys", "u {transcript} {codebook}", "claude-haiku-4-5-20251001",
            "t", [{"Code": "Q1"}], client,
        ):
            out.append(ev)
        return out

    events = asyncio.run(collect())
    types = [e["type"] for e in events]
    assert types == ["item", "item", "done"], f"unexpected sequence: {types}"
    print("OK bridge: forwards anthropic stream through asyncio.Queue")


if __name__ == "__main__":
    test_anthropic_stream_progressive()
    test_anthropic_cache_replay()
    test_anthropic_recovery_from_stringified_array()
    test_openai_stream_progressive()
    test_openai_final_text_fallback()
    test_groq_stream_jsonl()
    test_groq_filters_fences_and_garbage()
    test_groq_no_items_yields_error()
    test_ollama_stream_jsonl()
    test_ollama_done_reason_length_keeps_items()
    test_bridge_consumes_provider_stream()
    print("\nall provider stream smoke tests passed")
