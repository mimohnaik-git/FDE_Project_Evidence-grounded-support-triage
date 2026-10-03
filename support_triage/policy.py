"""Conservative demo business rules and deterministic evidence contract."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TriageResult(BaseModel):
    category: Literal["billing", "technical", "general", "escalation", "unsupported", "ambiguous"]
    priority: Literal["low", "medium", "high"]
    reasoning: str = Field(min_length=1)


class HumanDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["approve", "edit", "reject"]
    reviewer: str = Field(min_length=1)
    response: str = ""
    reason: str = Field(min_length=1)


def decision(triage, message):
    high_risk = bool(
        re.search(r"legal|lawyer|fraud|breach|security|disput|third time|outage|data loss", message, re.I)
    )
    requested = bool(re.search(r"human|supervisor|speak to.*(person|agent)|manual support", message, re.I))
    review = high_risk or requested or triage.category in {"escalation", "unsupported", "ambiguous"}
    queue = {
        "billing": "billing-support",
        "technical": "technical-support",
        "general": "customer-support",
    }.get(triage.category, "support-supervisor")
    priority = "high" if high_risk else triage.priority
    return dict(
        category=triage.category,
        priority=priority,
        assigned_queue=queue,
        handling_action="human_review" if review else "retrieve",
        routing_reason="High risk or human request" if high_risk or requested else triage.reasoning,
        human_review_required=review,
        sla_class={"high": "urgent-review", "medium": "standard", "low": "routine"}[priority],
    )


STOP = set("a an the i my me is are was to for of and on in it can do you your our please".split())


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def relevance(query, hit):
    words = tokens(query)
    return len(words & tokens(hit["title"] + " " + hit["text"])) / max(1, len(words))


# Topic anchors are company-KB scope, not customer classification labels.
SCOPE = {
    "billing-001": r"refund",
    "billing-002": r"payment|card|declin",
    "billing-003": r"subscription|billing cycle",
    "billing-004": r"invoic|receipt",
    "technical-001": r"login|log in|credentials|session|cookies",
    "technical-002": r"password|reset email",
    "technical-003": r"crash|startup",
    "technical-004": r"sync",
    "general-001": r"support|contact",
    "general-002": r"platform|product|tiers|project management",
}


def compatible(query, hit):
    pattern = SCOPE.get(hit.get("id"))
    return bool(pattern and re.search(pattern, query, re.IGNORECASE))


def sufficient(query, hits, threshold=0.35):
    return any(compatible(query, hit) and relevance(query, hit) >= threshold for hit in hits)


def grounded(draft, hits, evidence_status):
    if evidence_status != "sufficient" or not hits or not draft.strip():
        return False
    # Exact source passages make policy-number, negation and citation checks decidable.
    passages = {f"[{h['id']}] {h['text']}" for h in hits}
    lines = draft.split("\n\n")
    return bool(lines) and all(line in passages for line in lines)


def validate_human(value, state):
    item = HumanDecision.model_validate(value)
    if not item.reviewer.strip() or not item.reason.strip():
        raise ValueError("Reviewer and reason are required")
    if item.action == "edit" and not item.response.strip():
        raise ValueError("An edited response is required")
    if item.action == "approve" and state.get("grounding_status") != "passed":
        raise ValueError("Only a grounded draft can be approved; edit or reject this case")
    if item.action == "approve":
        # A pending checkpoint may predate a KB policy change.
        from support_triage.kb import articles

        corpus = {a["id"]: a["text"] for a in articles()}
        hits = state.get("kb_hits", [])
        if not hits or any(corpus.get(h["id"]) != h["text"] for h in hits):
            raise ValueError("Evidence changed; edit or reject the draft after reviewing current policy")
        if not grounded(state.get("draft_response", ""), hits, state.get("evidence_status")):
            raise ValueError("Draft no longer meets the grounding contract")
    return item
