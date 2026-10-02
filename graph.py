"""
graph.py
Builds the LangGraph StateGraph for the Customer Support Triage system.

Flow:

              ┌────────────────┐
              │  intake_agent   │
              └───────┬─────────┘
                       │ (conditional routing)
        ┌──────────────┼──────────────┬───────────────┐
        ▼              ▼              ▼                ▼
  billing_agent  technical_agent  general_agent   human_review_node
        │              │              │           (INTERRUPT BEFORE:
        │              │              │            pauses for a human
        └──────┬───────┴──────┬───────┘             supervisor's notes)
               ▼               ▼
               compose_agent ◄─┘
                   │
                  END

Only the escalation path passes through human_review_node, and the graph is
compiled with interrupt_before=["human_review_node"], so execution pauses
there until the Streamlit app resumes it with supervisor notes.

Which LLM backs every node (OpenAI / Anthropic / local Ollama) is resolved
once per build_graph() call via agents.build_agents() / llm_config.py.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents import TicketState, build_agents


def build_graph(provider: str | None = None, model: str | None = None):
    """
    Build and compile the triage graph.

    Returns (app, provider_name, model_name) — the resolved LLM backend is
    handed back so the UI can display exactly what a run will use.
    """
    nodes = build_agents(provider=provider, model=model)

    workflow = StateGraph(TicketState)

    workflow.add_node("intake_agent", nodes["intake_agent"])
    workflow.add_node("billing_agent", nodes["billing_agent"])
    workflow.add_node("technical_agent", nodes["technical_agent"])
    workflow.add_node("general_agent", nodes["general_agent"])
    workflow.add_node("human_review_node", nodes["human_review_node"])
    workflow.add_node("compose_agent", nodes["compose_agent"])

    workflow.set_entry_point("intake_agent")

    workflow.add_conditional_edges(
        "intake_agent",
        nodes["route_after_intake"],
        {
            "billing_agent": "billing_agent",
            "technical_agent": "technical_agent",
            "general_agent": "general_agent",
            "human_review_node": "human_review_node",
        },
    )

    workflow.add_edge("billing_agent", "compose_agent")
    workflow.add_edge("technical_agent", "compose_agent")
    workflow.add_edge("general_agent", "compose_agent")
    workflow.add_edge("human_review_node", "compose_agent")
    workflow.add_edge("compose_agent", END)

    checkpointer = MemorySaver()

    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review_node"],
    )

    return app, nodes["provider"], nodes["model"]


if __name__ == "__main__":
    # Quick manual smoke test: python graph.py
    from dotenv import load_dotenv

    load_dotenv()

    app, provider, model = build_graph()
    print(f"Using provider={provider} model={model}")
    config = {"configurable": {"thread_id": "test-thread-1"}}

    initial_state = {
        "ticket_id": "T-001",
        "customer_message": "This is the third time I'm writing! My subscription was "
        "charged twice and nobody has fixed it. I want a refund NOW or I'm cancelling "
        "and leaving a bad review.",
        "trace": [],
        "kb_hits": [],
        "needs_human": False,
        "human_notes": None,
    }

    result = app.invoke(initial_state, config=config)
    print("\n".join(result["trace"]))

    state = app.get_state(config)
    if state.next:
        print(f"\n⏸️  Graph paused before: {state.next}")
        app.update_state(config, {"human_notes": "Approve full refund, apologize for the delay."})
        result = app.invoke(None, config=config)
        print("\n".join(result["trace"][len(result["trace"]) - 2 :]))

    print("\n--- FINAL RESPONSE ---\n")
    print(result["final_response"])
