"""Persistent LangGraph workflow with structured human interrupts."""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from agents import TicketState, build_agents
from policy import validate_human

CHECKPOINT_PATH = Path(__file__).resolve().parent / "data" / "reviews.sqlite"


def build_graph(
    provider=None,
    model=None,
    embedding_provider=None,
    credentials=None,
    checkpoint_path=CHECKPOINT_PATH,
    **options,
):
    nodes = build_agents(provider, model, embedding_provider, credentials, **options)
    workflow = StateGraph(TicketState)
    for name in ("intake", "retrieve", "specialist", "validate", "review", "compose"):
        workflow.add_node(name, nodes[name])
    workflow.set_entry_point("intake")
    workflow.add_conditional_edges("intake", lambda s: "review" if s["human_review_required"] else "retrieve")
    workflow.add_conditional_edges(
        "retrieve", lambda s: "review" if s["human_review_required"] else "specialist"
    )
    workflow.add_edge("specialist", "validate")
    workflow.add_conditional_edges(
        "validate", lambda s: "review" if s["human_review_required"] else "compose"
    )
    workflow.add_edge("review", "compose")
    workflow.add_edge("compose", END)
    connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    graph = workflow.compile(checkpointer=SqliteSaver(connection))
    return graph, nodes["provider"], nodes["model"]


def pending_reviews(graph):
    # Public saver API supports listing all threads; graph history needs a thread ID.
    identifiers = {item.config["configurable"]["thread_id"] for item in graph.checkpointer.list(None)}
    snapshots = [
        graph.get_state({"configurable": {"thread_id": identifier}}) for identifier in sorted(identifiers)
    ]
    return [snapshot for snapshot in snapshots if snapshot.next == ("review",)]


def request_review(graph, config):
    snapshot = graph.get_state(config)
    if snapshot.next or snapshot.values.get("grounding_status") != "passed":
        raise ValueError("Only completed grounded drafts can be submitted for additional review")
    graph.update_state(
        config,
        {"human_review_required": True, "handling_action": "human_review", "final_response": ""},
        as_node="validate",
    )
    return graph.invoke(None, config)


def resume_review(graph, config, value):
    state = graph.get_state(config)
    if state.next != ("review",):
        raise ValueError("Ticket is not awaiting review")
    validate_human(value, state.values)
    return graph.invoke(Command(resume=value), config)
