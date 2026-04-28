"""Ad-hoc smoke checks for streaming parsers. Run with:
    PYTHONPATH=. python tests/_stream_parse_smoke.py
"""
from talktrace_ai.utils.llm_analysis._stream_parse import (
    find_array_start,
    extract_new_items,
    normalize_item,
    parse_jsonl_line,
)


def test_progressive_array_walk():
    full = ('{"analysis": [{"#": 1, "Sprecher": "L", "Shortcode": "Q1", "Impuls": "Hi"}, '
            '{"#": 2, "Sprecher": "S01", "Shortcode": "A1", "Impuls": "Yo"}]}')
    start = find_array_start(full, "analysis")
    assert start > 0, f"array start should be positive, got {start}"
    items, npos = extract_new_items(full, start)
    assert len(items) == 2, f"expected 2 items, got {len(items)}: {items}"
    assert items[0]["Shortcode"] == "Q1"
    assert items[1]["Sprecher"] == "S01"
    print("OK progressive array walk: 2 items extracted")


def test_progressive_streaming_growth():
    """Simulate buffer growth across deltas; only emit completed items."""
    pieces = [
        '{"analysis":',
        ' [{"#": 1, "Sprecher": "L",',
        ' "Shortcode": "Q1", "Impuls": "Hi"}',
        ', {"#": 2, "Sprecher": "S01"',
        ', "Shortcode": "A1", "Impuls": "Yo"}',
        ', {"#": 3, "Sprecher": "S02", "Shortcode": "B1", "Impuls": "Sup"}',
        ']}',
    ]
    buf = ""
    pos = -1
    next_pos = 0
    emitted = []
    for chunk in pieces:
        buf += chunk
        if pos < 0:
            pos = find_array_start(buf, "analysis")
            if pos >= 0:
                next_pos = pos
        if pos < 0:
            continue
        new_items, next_pos = extract_new_items(buf, next_pos)
        emitted.extend(new_items)
    assert len(emitted) == 3, f"expected 3 items, got {len(emitted)}"
    nums = [it["#"] for it in emitted]
    assert nums == [1, 2, 3], f"sequence wrong: {nums}"
    print("OK progressive streaming growth: 3 items in order")


def test_normalize_item_backfills():
    obj = {"Shortcode": "Q1", "Impuls": "hello"}
    n = normalize_item(obj)
    assert n is not None
    assert n["Sprecher"] == ""
    assert n["#"] == 0
    print("OK normalize backfills missing fields")


def test_jsonl_line():
    line = '{"#": 5, "Sprecher": "S02", "Shortcode": "B2", "Impuls": "Hey"}'
    item = parse_jsonl_line(line)
    assert item is not None
    assert item["Shortcode"] == "B2"
    assert parse_jsonl_line("not-json") is None
    assert parse_jsonl_line("```json") is None
    assert parse_jsonl_line("") is None
    # Trailing comma tolerance
    item2 = parse_jsonl_line('{"#": 1, "Sprecher": "L", "Shortcode": "X", "Impuls": "z"},')
    assert item2 is not None and item2["Shortcode"] == "X"
    print("OK jsonl line parsing handles fences/empty/trailing comma")


def test_async_stream_bridge():
    """Smoke-test that the bridge yields events through asyncio in order."""
    import asyncio
    from talktrace_ai.utils.llm_analysis._stream_bridge import async_stream

    def producer():
        yield {"type": "item", "data": {"i": 1}}
        yield {"type": "item", "data": {"i": 2}}
        yield {"type": "done"}

    async def main():
        out = []
        async for ev in async_stream(producer):
            out.append(ev)
        return out

    events = asyncio.run(main())
    assert [e["type"] for e in events] == ["item", "item", "done"]
    print("OK async_stream forwards events in order")


def test_async_stream_bridge_error():
    """An exception inside the producer becomes an error event."""
    import asyncio
    from talktrace_ai.utils.llm_analysis._stream_bridge import async_stream

    def boom():
        yield {"type": "item", "data": {"i": 1}}
        raise RuntimeError("boom")

    async def main():
        out = []
        async for ev in async_stream(boom):
            out.append(ev)
        return out

    events = asyncio.run(main())
    assert events[0]["type"] == "item"
    assert events[-1]["type"] == "error"
    assert "boom" in events[-1]["message"]
    print("OK async_stream surfaces producer exceptions")


if __name__ == "__main__":
    test_progressive_array_walk()
    test_progressive_streaming_growth()
    test_normalize_item_backfills()
    test_jsonl_line()
    test_async_stream_bridge()
    test_async_stream_bridge_error()
    print("\nall stream-parse smoke tests passed")
