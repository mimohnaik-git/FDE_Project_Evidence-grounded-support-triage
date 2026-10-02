"""
agents.py
State schema + agent node functions for the Customer Support Triage graph.

Pipeline:
  intake_agent        -> classifies category + priority
  billing_agent        \
  technical_agent       > specialist drafts a response, using Chroma KB context
  general_agent         /
  human_review_node   -> only reached for 'escalation' category; graph pauses here
  compose_agent       -> formats the final customer-facing reply

Which chat LLM backs all of this (OpenAI / Anthropic / local Ollama) is
resolved once per graph build by `build_agents()`, using llm_config.py.
Node functions are built as closures over that resolved provider/model so
every node in a given run uses the same backend.
"""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

import kb
from llm_config import get_chat_model


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class TicketState(TypedDict):
    ticket_id: str
    customer_message: str
    category: Optional[str]          # billing | technical | general | escalation
    priority: Optional[str]          # low | medium | high
    reasoning: Optional[str]
    kb_hits: List[dict]
    draft_response: Optional[str]
    final_response: Optional[str]
    trace: List[str]
    needs_human: bool
    human_notes: Optional[str]


# ---------------------------------------------------------------------------
# Structured output schema for triage classification
# ---------------------------------------------------------------------------
class TriageResult(BaseModel):
    category: Literal["billing", "technical", "general", "escalation"] = Field(
        description="Best-fit category for routing this ticket."
    )
    priority: Literal["low", "medium", "high"] = Field(
        description="Urgency of the ticket based on customer tone and impact."
    )
    reasoning: str = Field(description="One-sentence justification for the routing decision.")


def build_agents(provider: str | None = None, model: str | None = None):
    """
    Build all graph node functions bound to one resolved LLM provider/model.

    Returns a dict of node functions plus "provider" and "model" (the
    resolved names, for display in the UI).
    """
    # Resolve once up front so every node in this run uses the same backend,
    # and so we can show the user which provider/model is active before any
    # LLM calls actually happen.
    _, resolved_provider, resolved_model = get_chat_model(provider=provider, model=model)

    def _llm(temperature: float = 0.0):
        chat, _, _ = get_chat_model(provider=resolved_provider, model=resolved_model, temperature=temperature)
        return chat

    # -----------------------------------------------------------------
    # Nodes
    # -----------------------------------------------------------------
    def intake_agent(state: TicketState) -> TicketState:
        """Classifies the ticket into a category + priority."""
        structured_llm = _llm().with_structured_output(TriageResult)

        system = (
            "You are an intake triage agent for a customer support system. "
            "Classify the ticket into exactly one category:\n"
            "- billing: payments, refunds, invoices, subscriptions\n"
            "- technical: bugs, login issues, crashes, sync problems\n"
            "- general: hours, product info, anything not billing/technical\n"
            "- escalation: angry/threatening customers, legal threats, repeated unresolved "
            "issues, requests to cancel due to a bad experience, or anything needing a human.\n"
            "Also assign a priority (low/medium/high) based on urgency and customer sentiment."
        )

        result: TriageResult = structured_llm.invoke(
            [("system", system), ("human", state["customer_message"])]
        )

        trace = state.get("trace", [])
        trace.append(
            f"🔎 Intake Agent → category: **{result.category}**, priority: **{result.priority}**"
        )

        return {
            **state,
            "category": result.category,
            "priority": result.priority,
            "reasoning": result.reasoning,
            "trace": trace,
            "needs_human": result.category == "escalation",
        }

    def _specialist_agent(state: TicketState, category: str, persona: str) -> TicketState:
        """Shared logic for billing/technical/general specialist agents."""
        hits = kb.retrieve(state["customer_message"], category=category, k=2)

        context = "\n\n".join(f"[{h['title']}]: {h['text']}" for h in hits) or "No KB articles found."

        system = (
            f"You are the {persona}, a specialist customer support agent. "
            "Use the knowledge base context below to write a helpful, accurate, empathetic draft "
            "response to the customer. Be concise (3-5 sentences). Do not invent policies not "
            "present in the context.\n\nKNOWLEDGE BASE CONTEXT:\n" + context
        )

        response = _llm(temperature=0.3).invoke(
            [("system", system), ("human", state["customer_message"])]
        )

        trace = state.get("trace", [])
        trace.append(f"🛠️ {persona} → drafted response using {len(hits)} KB article(s)")

        return {
            **state,
            "kb_hits": hits,
            "draft_response": response.content,
            "trace": trace,
        }

    def billing_agent(state: TicketState) -> TicketState:
        return _specialist_agent(state, "billing", "Billing Agent")

    def technical_agent(state: TicketState) -> TicketState:
        return _specialist_agent(state, "technical", "Technical Support Agent")

    def general_agent(state: TicketState) -> TicketState:
        return _specialist_agent(state, "general", "General Inquiries Agent")

    def human_review_node(state: TicketState) -> TicketState:
        """
        Reached only for 'escalation' tickets. The graph is compiled with
        interrupt_before=['human_review_node'], so execution pauses BEFORE this node runs
        and resumes here once a human has supplied notes via the Streamlit UI.
        """
        trace = state.get("trace", [])
        human_notes = state.get("human_notes") or "(No notes provided by human reviewer.)"

        system = (
            "You are drafting an escalation response on behalf of a human support supervisor. "
            "Incorporate the supervisor's notes into a calm, empathetic, professional reply "
            "to the customer. Acknowledge their frustration and be specific about next steps."
        )
        human_input = (
            f"Customer message:\n{state['customer_message']}\n\n"
            f"Supervisor notes:\n{human_notes}"
        )

        response = _llm(temperature=0.3).invoke([("system", system), ("human", human_input)])

        trace.append("🧑‍💼 Human Review → supervisor notes incorporated into escalation response")

        return {
            **state,
            "draft_response": response.content,
            "trace": trace,
        }

    def compose_agent(state: TicketState) -> TicketState:
        """Final formatting pass: adds greeting/sign-off and consistent tone."""
        system = (
            "You are the final response composer for a customer support system. "
            "Take the draft response and format it into a polished, ready-to-send email: "
            "brief greeting, the core message, and a friendly sign-off from 'The Support Team'. "
            "Keep the substance of the draft unchanged."
        )

        response = _llm(temperature=0.2).invoke(
            [("system", system), ("human", state.get("draft_response", ""))]
        )

        trace = state.get("trace", [])
        trace.append("✅ Compose Agent → finalized customer-facing reply")

        return {
            **state,
            "final_response": response.content,
            "trace": trace,
        }

    def route_after_intake(state: TicketState) -> str:
        category = state.get("category", "general")
        return {
            "billing": "billing_agent",
            "technical": "technical_agent",
            "general": "general_agent",
            "escalation": "human_review_node",
        }.get(category, "general_agent")

    return {
        "intake_agent": intake_agent,
        "billing_agent": billing_agent,
        "technical_agent": technical_agent,
        "general_agent": general_agent,
        "human_review_node": human_review_node,
        "compose_agent": compose_agent,
        "route_after_intake": route_after_intake,
        "provider": resolved_provider,
        "model": resolved_model,
    }
