"""Independent deterministic fixtures for CI; never advertised as model quality."""

import re

import numpy as np

from kb import articles
from policy import TriageResult, relevance, tokens


class FakeChat:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        text = messages[-1][1].lower()
        if re.search(r"lawyer|fraud|security|third time|outage|human|supervisor", text):
            category, priority = "escalation", "high"
        elif re.search(r"refund|payment|card|cancel|invoice|subscription|charged", text):
            category, priority = "billing", "medium"
        elif re.search(r"login|password|crash|sync|credentials", text):
            category, priority = "technical", "medium"
        elif re.search(r"support hours|platform|product|tiers|plan", text):
            category, priority = "general", "low"
        else:
            category, priority = "unsupported", "low"
        return TriageResult(category=category, priority=priority, reasoning="Offline lexical fixture")


def fake_retrieve(query, category=None, k=2, **kwargs):
    rows = [a for a in articles() if not category or a["category"] == category]
    return sorted(rows, key=lambda a: (-relevance(query, a), a["id"]))[:k]


class FakeEmbedding:
    """Fixed signed feature hashing: no pretrained model or network download."""

    def __call__(self, input):
        import hashlib

        vectors = []
        for text in input:
            vector = np.zeros(128)
            for word in tokens(text):
                value = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16)
                vector[value % 128] += 1 if value % 2 else -1
            vectors.append((vector / max(np.linalg.norm(vector), 1)).tolist())
        return vectors

    def name(self):
        return "offline-feature-hash-v1"

    def embed_query(self, input):
        return self(input)

    def embed_documents(self, input):
        return self(input)
