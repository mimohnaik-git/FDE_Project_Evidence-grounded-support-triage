"""
embedding_config.py
--------------------
Picks which embedding function Chroma uses for the knowledge base.

    - "openai" (paid): text-embedding-3-small, needs OPENAI_API_KEY.
    - "local"  (free): all-MiniLM-L6-v2, runs fully on-device via chromadb's
      bundled ONNX runtime. No API key, no per-call cost. The ~80MB model is
      downloaded once (from chroma-onnx-models.s3.amazonaws.com) and cached
      at ~/.cache/chroma, then works fully offline. If your network blocks
      that host, set EMBEDDING_PROVIDER=openai instead.

Default is "auto": use OpenAI if a key is present (better retrieval
quality), otherwise fall back to the free local model automatically — so
the knowledge base works either way with zero configuration.
"""

from __future__ import annotations

import os

from chromadb.utils import embedding_functions

DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "local": "all-MiniLM-L6-v2",
}


def resolve_embedding_provider(preferred: str | None = None) -> str:
    preferred = preferred or os.getenv("EMBEDDING_PROVIDER", "auto")
    if preferred != "auto":
        return preferred
    return "openai" if os.getenv("OPENAI_API_KEY") else "local"


def get_embedding_function(provider: str | None = None):
    """
    Returns (embedding_function, provider_name, model_name).

    Note: the Chroma collection name is provider-specific (see kb.py), so
    switching providers doesn't try to mix incompatible embedding vectors
    in one collection — each provider gets its own collection, seeded
    independently from data/kb_articles.json.
    """
    provider = resolve_embedding_provider(provider)

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set.")
        fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=DEFAULT_MODELS["openai"],
        )
        return fn, provider, DEFAULT_MODELS["openai"]

    if provider == "local":
        fn = embedding_functions.DefaultEmbeddingFunction()
        return fn, provider, DEFAULT_MODELS["local"]

    raise ValueError(f"Unknown EMBEDDING_PROVIDER '{provider}'. Use openai, local, or auto.")
