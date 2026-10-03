"""Chroma index versioned by source content, provider and embedding model."""

import hashlib
import json
from pathlib import Path

import chromadb

from embedding_config import get_embedding_function

ROOT = Path(__file__).resolve().parent
KB_JSON_PATH = ROOT / "data" / "kb_articles.json"
CHROMA_PATH = ROOT / "chroma_db"


def articles(path=KB_JSON_PATH):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if len({a["id"] for a in rows}) != len(rows):
        raise ValueError("Duplicate KB identifiers")
    return rows


def fingerprint(rows):
    canonical = json.dumps(sorted(rows, key=lambda a: a["id"]), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def get_collection(provider=None, credentials=None, *, client=None, embedding=None, source=KB_JSON_PATH):
    fn, resolved, model = embedding or get_embedding_function(provider, credentials)
    rows = articles(source)
    version = fingerprint(rows)
    identity = hashlib.sha256(f"{resolved}:{model}:{version}".encode()).hexdigest()[:24]
    client = client if client is not None else chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=f"support_{identity}",
        embedding_function=fn,
        metadata={
            "source_fingerprint": version,
            "provider": resolved,
            "model": model,
            "hnsw:space": "cosine",
        },
    )
    # Upsert also repairs an interrupted/partial seed. Immutable version names avoid stale content.
    if collection.count() != len(rows):
        collection.upsert(
            ids=[a["id"] for a in rows],
            documents=[a["text"] for a in rows],
            metadatas=[{"title": a["title"], "category": a["category"]} for a in rows],
        )
    return collection


def retrieve(query, category=None, k=2, provider=None, credentials=None, **options):
    collection = get_collection(provider, credentials, **options)
    result = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        where={"category": category} if category else None,
    )
    return [
        dict(id=i, text=text, title=meta["title"], category=meta["category"], distance=distance)
        for i, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
