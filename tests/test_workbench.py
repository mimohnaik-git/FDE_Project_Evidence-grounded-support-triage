import json
from pathlib import Path

import chromadb
import pytest

from agents import build_agents
from embedding_config import resolve_embedding_provider
from evaluation.adapters import adapt, sample
from evaluation.offline import FakeChat, FakeEmbedding, fake_retrieve
from graph import build_graph, pending_reviews, request_review, resume_review
from kb import articles, fingerprint, get_collection, retrieve
from llm_config import resolve_provider
from policy import TriageResult, decision, grounded, sufficient
from runtime import Credentials, safe_failure


@pytest.mark.parametrize(
    "preferred,credentials,expected",
    [
        ("auto", Credentials(), "ollama"),
        ("auto", Credentials("secret"), "openai"),
        ("auto", Credentials("", "secret"), "anthropic"),
        ("ollama", Credentials("secret"), "ollama"),
    ],
)
def test_provider(preferred, credentials, expected):
    assert resolve_provider(preferred, credentials) == expected


@pytest.mark.parametrize("preferred,expected", [("local", "local"), ("openai", "openai"), ("auto", "local")])
def test_embedding_resolution(preferred, expected):
    assert resolve_embedding_provider(preferred, Credentials()) == expected


def test_invalid_providers():
    with pytest.raises(ValueError):
        resolve_provider("wrong")
    with pytest.raises(ValueError):
        resolve_embedding_provider("wrong")


@pytest.mark.parametrize(
    "category,queue,review",
    [
        ("billing", "billing-support", False),
        ("technical", "technical-support", False),
        ("general", "customer-support", False),
        ("unsupported", "support-supervisor", True),
        ("ambiguous", "support-supervisor", True),
        ("escalation", "support-supervisor", True),
    ],
)
def test_policy(category, queue, review):
    result = decision(TriageResult(category=category, priority="medium", reasoning="test"), "hello")
    assert result["assigned_queue"] == queue
    assert result["human_review_required"] == review


@pytest.mark.parametrize("message", ["legal dispute", "security breach", "I want a human", "outage"])
def test_policy_override(message):
    result = decision(TriageResult(category="general", priority="low", reasoning="test"), message)
    assert result["human_review_required"]


def test_embedding_propagation_and_immutable_trace():
    captured = {}

    def retriever(query, **kwargs):
        captured.update(kwargs)
        return []

    nodes = build_agents(
        chat=FakeChat(), retriever=retriever, embedding_provider="local", credentials=Credentials("secret")
    )
    original = {"customer_message": "refund policy", "trace": ["original"]}
    output = nodes["intake"](original)
    nodes["retrieve"]({**original, **output})
    assert captured["provider"] == "local"
    assert captured["credentials"].openai == "secret"
    assert original["trace"] == ["original"]


def test_kb_fingerprint_and_invalidation(tmp_path):
    rows = articles()
    source = tmp_path / "kb.json"
    source.write_text(json.dumps(rows))
    client = chromadb.EphemeralClient()
    embedding = (FakeEmbedding(), "stub", "fixed-v1")
    old = get_collection(client=client, embedding=embedding, source=source)
    rows[0]["text"] = "Changed refund policy"
    source.write_text(json.dumps(rows))
    new = get_collection(client=client, embedding=embedding, source=source)
    assert old.name != new.name
    assert fingerprint(rows) == fingerprint(list(reversed(rows)))
    assert "Changed refund policy" in new.get(ids=["billing-001"])["documents"]


def test_chroma_filter_and_ids(tmp_path):
    hits = retrieve(
        "refund",
        category="billing",
        client=chromadb.EphemeralClient(),
        embedding=(FakeEmbedding(), "stub", "fixed-v1"),
    )
    assert len(hits) == 2
    assert all(h["category"] == "billing" and h["id"].startswith("billing") for h in hits)


def test_evidence_gate():
    assert sufficient("refund policy within 14 days", articles()[:1])
    assert not sufficient("write a poem about Jupiter", articles())
    assert not sufficient("refund", [])


@pytest.mark.parametrize(
    "draft,status,expected",
    [
        ("[billing-001] EXACT", "sufficient", True),
        ("[billing-001] Wrong 99 days", "sufficient", False),
        ("[wrong] EXACT", "sufficient", False),
        ("[billing-001] EXACT", "insufficient", False),
    ],
)
def test_grounding(draft, status, expected):
    assert grounded(draft, [{"id": "billing-001", "text": "EXACT"}], status) == expected


@pytest.fixture
def graph(tmp_path):
    value, _, _ = build_graph(
        chat=FakeChat(), retriever=fake_retrieve, checkpoint_path=tmp_path / "review.sqlite"
    )
    yield value
    value.checkpointer.conn.close()


@pytest.mark.parametrize(
    "message", ["What are support hours?", "refund policy within 14 days", "password reset email 15 minutes"]
)
def test_happy_paths(graph, message):
    result = graph.invoke(
        {"ticket_id": "happy", "customer_message": message, "trace": []},
        {"configurable": {"thread_id": "happy"}},
    )
    assert result["grounding_status"] == "passed"
    assert result["final_response"].startswith("Hello,")


@pytest.mark.parametrize("action", ["edit", "reject"])
def test_review_decisions(graph, action):
    config = {"configurable": {"thread_id": action}}
    result = graph.invoke({"ticket_id": action, "customer_message": "I want a human", "trace": []}, config)
    assert not result["final_response"]
    value = dict(action=action, reviewer="Operator", reason="Reviewed", response="We will contact you.")
    result = resume_review(graph, config, value)
    assert result["human_decision"]["action"] == action
    assert bool(result["final_response"]) == (action == "edit")


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"action": "invalid"},
        dict(action="edit", reviewer="A", reason="B", response=" "),
        dict(action="reject", reviewer=" ", reason="B"),
        dict(action="approve", reviewer="A", reason="B"),
    ],
)
def test_invalid_review_preserves_pending(graph, value):
    config = {"configurable": {"thread_id": "invalid"}}
    graph.invoke({"ticket_id": "invalid", "customer_message": "write a poem", "trace": []}, config)
    with pytest.raises(ValueError):
        resume_review(graph, config, value)
    assert graph.get_state(config).next == ("review",)


def test_approve_grounded_review(graph):
    config = {"configurable": {"thread_id": "approve"}}
    graph.invoke({"ticket_id": "approve", "customer_message": "refund within 14 days", "trace": []}, config)
    request_review(graph, config)
    assert graph.get_state(config).next == ("review",)
    result = resume_review(graph, config, dict(action="approve", reviewer="A", reason="Reviewed"))
    assert result["final_response"]
    assert result["human_decision"]["action"] == "approve"


def test_restart_persistence(tmp_path):
    path = tmp_path / "persist.sqlite"
    config = {"configurable": {"thread_id": "durable"}}
    first, _, _ = build_graph(chat=FakeChat(), retriever=fake_retrieve, checkpoint_path=path)
    first.invoke({"ticket_id": "durable", "customer_message": "security breach", "trace": []}, config)
    first.checkpointer.conn.close()
    second, _, _ = build_graph(chat=FakeChat(), retriever=fake_retrieve, checkpoint_path=path)
    assert second.get_state(config).next == ("review",)
    assert len(pending_reviews(second)) == 1
    result = resume_review(second, config, dict(action="reject", reviewer="A", reason="Manual"))
    assert result["handling_action"] == "manual_handling"
    assert not pending_reviews(second)
    second.checkpointer.conn.close()


def test_grounding_failure_routes_to_review(graph, monkeypatch):
    import agents

    monkeypatch.setattr(agents, "grounded", lambda *args: False)
    config = {"configurable": {"thread_id": "failure"}}
    result = graph.invoke(
        {"ticket_id": "failure", "customer_message": "refund within 14 days", "trace": []}, config
    )
    assert result["grounding_status"] == "failed"
    assert graph.get_state(config).next == ("review",)
    assert not result["final_response"]


def test_sanitized_failure(caplog):
    message = safe_failure(RuntimeError("sk-private-secret customer@example.com"))
    assert "sk-private-secret" not in message + caplog.text
    assert "customer@example.com" not in message + caplog.text
    assert "RuntimeError" in caplog.text


def test_safe_ui_output():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "unsafe_allow_html" not in source
    assert "os.environ[" not in source
    assert "st.text(" in source


def test_adapters_no_label_leakage():
    cases = adapt(
        "bitext",
        [
            {
                "instruction": "where is invoice?",
                "intent": "get_invoice",
                "response": "SECRET ANSWER",
                "category": "LABEL",
            }
        ],
    )
    assert cases[0]["message"] == "where is invoice?"
    assert "SECRET" not in cases[0]["message"]
    assert sample(cases, seed=42) == sample(cases, seed=42)
    assert adapt("clinc", [["poem", "oos"], ["refund", "refund"]])[0]["category"] == "unsupported"


def test_evaluation_safety():
    from evaluation.run import run

    result = run()
    assert result["false_safe_rate"] == 0
    assert result["grounding_violations"] == 0


def test_unrelated_cancellation_is_not_subscription_evidence(graph):
    message = "is there a fee for cancelling a flight i've booked"
    result = graph.invoke(
        {"ticket_id": "flight", "customer_message": message, "trace": []},
        {"configurable": {"thread_id": "flight"}},
    )
    assert result["human_review_required"]
    assert result["handling_action"] == "human_review"
    assert not result["final_response"]


def test_calibration_threshold_is_current():
    from evaluation.calibrate import calibrate

    assert calibrate()["threshold"] == 0.35


def test_fabricated_article_text_fails_grounding(tmp_path):
    def tampered(query, **kwargs):
        return [
            {
                "id": "billing-001",
                "title": "Refund Policy",
                "category": "billing",
                "text": "Refund within 999 days of purchase",
            }
        ]

    graph, _, _ = build_graph(
        chat=FakeChat(), retriever=tampered, checkpoint_path=tmp_path / "tampered.sqlite"
    )
    result = graph.invoke(
        {"ticket_id": "tampered", "customer_message": "refund within days of purchase", "trace": []},
        {"configurable": {"thread_id": "tampered"}},
    )
    assert result["grounding_status"] == "failed"
    assert not result["final_response"]
    graph.checkpointer.conn.close()


def test_session_credentials_do_not_escape(tmp_path, monkeypatch):
    import os

    key = "private-session-credential-sentinel"
    monkeypatch.setenv("OPENAI_API_KEY", "environment-sentinel")
    path = tmp_path / "credentials.sqlite"
    graph, _, _ = build_graph(
        chat=FakeChat(), retriever=fake_retrieve, credentials=Credentials(key), checkpoint_path=path
    )
    graph.invoke(
        {"ticket_id": "isolated", "customer_message": "refund policy", "trace": []},
        {"configurable": {"thread_id": "isolated"}},
    )
    graph.checkpointer.conn.close()
    assert key.encode() not in path.read_bytes()
    assert os.environ["OPENAI_API_KEY"] == "environment-sentinel"
    assert key not in repr(Credentials(key))


def test_approval_rejects_stale_policy(graph, monkeypatch):
    import kb

    config = {"configurable": {"thread_id": "stale"}}
    graph.invoke({"ticket_id": "stale", "customer_message": "refund within 14 days", "trace": []}, config)
    request_review(graph, config)
    changed = [{**a, "text": "Changed refund terms"} if a["id"] == "billing-001" else a for a in articles()]
    monkeypatch.setattr(kb, "articles", lambda: changed)
    with pytest.raises(ValueError):
        resume_review(graph, config, dict(action="approve", reviewer="A", reason="Reviewed"))
    assert graph.get_state(config).next == ("review",)
