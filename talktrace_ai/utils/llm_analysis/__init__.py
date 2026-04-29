"""Per-provider LLM analysis pipelines.

Public entry points (one per provider). Each takes a system prompt, user
prompt, model id, transcript text, codebook, plus an SDK client (or api_key
for Ollama) and returns a JSON string of coded impulses.

To add a new provider:
    1. Create ``utils/llm_analysis/<name>.py`` with a ``llm_analysis_<name>``
       function. Use one of the existing modules as a template.
    2. Add the import + re-export below.
    3. Wire it into the ConfigManager + ``handlers/server_body.py`` provider
       routing logic.
"""
from .groq import llm_analysis_groq, llm_analysis_groq_stream
from .openai import llm_analysis_openai, llm_analysis_openai_stream
from .anthropic import llm_analysis_anthropic, llm_analysis_anthropic_stream
from .ollama import llm_analysis_ollama, llm_analysis_ollama_stream
from ._stream_bridge import async_stream

__all__ = [
    "llm_analysis_groq",
    "llm_analysis_openai",
    "llm_analysis_anthropic",
    "llm_analysis_ollama",
    "llm_analysis_groq_stream",
    "llm_analysis_openai_stream",
    "llm_analysis_anthropic_stream",
    "llm_analysis_ollama_stream",
    "async_stream",
]
