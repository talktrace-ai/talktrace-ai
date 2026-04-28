"""tiktoken-based token counting + context-bucket helpers.

Only Ollama actually inspects tokens at runtime; the helpers are reused for
context-window bucketing. Keep them isolated so the rest of the pipeline
doesn't pay the cost of importing tiktoken when not needed.
"""
import tiktoken


_TIKTOKEN_ENCODING = None


def _get_encoding():
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODING


def _count_tokens(text):
    try:
        return len(_get_encoding().encode(text or ""))
    except Exception:
        return max(1, len(text or "") // 4)


_CTX_BUCKETS = (8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)


def _bucket_ctx(n):
    for b in _CTX_BUCKETS:
        if n <= b:
            return b
    return _CTX_BUCKETS[-1]


# Per-model context/output budgets for Ollama (local or cloud).
# Interpreted as MAX caps; actual num_ctx/num_predict are computed reactively
# from input token count in llm_analysis_ollama.
