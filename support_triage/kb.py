"""
kb.py
Chroma DB setup for the Customer Support Triage demo.

- Creates/loads a persistent Chroma collection at ./chroma_db
- Seeds it with sample knowledge-base articles (data/kb_articles.json) on first run
- Exposes retrieve() for the specialist agents to pull relevant context

Which embedding model is used (paid OpenAI vs free local) is decided by
embedding_config.py. Each provider gets its own collection name so
switching providers never mixes incompatible embedding vectors together.
"""

import json
import os

import chromadb

from embedding_config import get_embedding_function

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
KB_JSON_PATH = os.path.join(os.path.dirname(__file__), "data", "kb_articles.json")


def _get_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(provider: str | None = None):
    """Return the Chroma collection for the resolved embedding provider,
    seeding it with KB articles if empty."""
    embedding_fn, provider_name, model_name = get_embedding_function(provider)
    collection_name = f"support_kb_{provider_name}"

    client = _get_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    if collection.count() == 0:
        with open(KB_JSON_PATH, "r") as f:
            articles = json.load(f)

        try:
            collection.add(
                ids=[a["id"] for a in articles],
                documents=[a["text"] for a in articles],
                metadatas=[{"category": a["category"], "title": a["title"]} for a in articles],
            )
        except Exception as exc:
            _raise_friendly_embedding_error(provider_name, exc)

    return collection


def _raise_friendly_embedding_error(provider_name: str, exc: Exception) -> None:
    if provider_name == "local":
        raise RuntimeError(
            "Couldn't download the free local embedding model (needs one-time internet "
            "access to chroma-onnx-models.s3.amazonaws.com, then works offline). If this "
            "network blocks that host, set EMBEDDING_PROVIDER=openai and supply an "
            f"OPENAI_API_KEY instead. Original error: {exc}"
        ) from exc
    raise RuntimeError(f"Embedding call failed for provider '{provider_name}': {exc}") from exc


def retrieve(query: str, category: str | None = None, k: int = 2, provider: str | None = None):
    """
    Retrieve top-k relevant KB snippets for a query, optionally filtered by category
    ('billing', 'technical', 'general').

    Returns a list of dicts: {"title": ..., "text": ..., "category": ..., "score": ...}
    """
    collection = get_collection(provider)
    where_filter = {"category": category} if category else None

    try:
        results = collection.query(
            query_texts=[query],
            n_results=k,
            where=where_filter,
        )
    except Exception as exc:
        _raise_friendly_embedding_error(resolve_active_provider(provider), exc)

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(docs)

    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "title": meta.get("title", ""),
                "text": doc,
                "category": meta.get("category", ""),
                "score": dist,
            }
        )
    return hits


def resolve_active_provider(provider: str | None = None) -> str:
    _, provider_name, _ = get_embedding_function(provider)
    return provider_name


def reset_kb(provider: str | None = None):
    """Utility to wipe and re-seed the collection (useful during development)."""
    _, provider_name, _ = get_embedding_function(provider)
    client = _get_client()
    try:
        client.delete_collection(f"support_kb_{provider_name}")
    except Exception:
        pass
    return get_collection(provider)


if __name__ == "__main__":
    # Quick manual test: python kb.py
    from dotenv import load_dotenv

    load_dotenv()
    col = get_collection()
    provider_name = resolve_active_provider()
    print(f"Collection for provider '{provider_name}' has {col.count()} documents.")
    for hit in retrieve("my card keeps getting declined", category="billing"):
        print(f"- [{hit['title']}] {hit['text'][:80]}...")
