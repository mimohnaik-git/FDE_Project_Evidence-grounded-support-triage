"""Operator UI; no credentials in os.environ or checkpoint state."""

import uuid

import streamlit as st

from embedding_config import resolve_embedding_provider
from graph import build_graph, pending_reviews, request_review, resume_review
from runtime import Credentials, safe_failure

st.set_page_config(page_title="Evidence-Grounded Support Triage", layout="wide")
st.title("Evidence-Grounded Support Triage Workbench")
st.caption("Operator drafts only. Sending responses and account actions remain human-controlled.")
provider = st.sidebar.selectbox("Chat provider", ["auto", "openai", "anthropic", "ollama"])
embedding = st.sidebar.selectbox("Embedding provider", ["auto", "local", "openai"])
model = st.sidebar.text_input("Model override")
base = Credentials.from_environment()
credentials = Credentials(
    st.sidebar.text_input("OpenAI key", type="password") or base.openai,
    st.sidebar.text_input("Anthropic key", type="password") or base.anthropic,
)
# Session-owned cache; changes to credentials rebuild clients without exposing key values.
signature = (provider, embedding, model, credentials)
try:
    if st.session_state.get("signature") != signature:
        old = st.session_state.get("graph")
        if old:
            old.checkpointer.conn.close()
        graph, resolved, active_model = build_graph(provider, model or None, embedding, credentials)
        st.session_state.update(graph=graph, signature=signature, active=(resolved, active_model))
    graph = st.session_state.graph
    st.sidebar.write("Active chat:", *st.session_state.active)
    st.sidebar.write("Active embeddings:", resolve_embedding_provider(embedding, credentials))
except Exception as exc:
    st.error(safe_failure(exc))
    st.stop()

try:
    reviews = pending_reviews(graph)
    choices = {s.config["configurable"]["thread_id"]: s for s in reviews}
    selected = st.sidebar.selectbox("Pending reviews (persist across restart)", [""] + list(choices))
    if selected:
        st.session_state["thread_id"] = selected
except Exception as exc:
    st.error(safe_failure(exc))

message = st.text_area("Customer ticket")
if st.button("Run triage", type="primary"):
    if not message.strip():
        st.warning("Enter a ticket.")
    else:
        identifier = str(uuid.uuid4())
        config = {"configurable": {"thread_id": identifier}}
        try:
            graph.invoke({"ticket_id": identifier, "customer_message": message, "trace": []}, config)
            st.session_state["thread_id"] = identifier
            st.rerun()
        except Exception as exc:
            st.error(safe_failure(exc))

if st.session_state.get("thread_id"):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    snapshot = graph.get_state(config)
    result = snapshot.values
    st.write("Ticket ID:", st.session_state.thread_id)
    st.text(result.get("customer_message", ""))
    for field in (
        "provider",
        "model",
        "embedding_provider",
        "category",
        "priority",
        "assigned_queue",
        "handling_action",
        "sla_class",
        "routing_reason",
        "evidence_status",
        "grounding_status",
    ):
        st.text(f"{field}: {result.get(field, 'pending')}")
    st.subheader("Workflow trace")
    st.text("\n".join(result.get("trace", [])))
    with st.expander("Retrieved evidence", expanded=True):
        for hit in result.get("kb_hits", []):
            st.text(f"[{hit['id']}] {hit['title']}\n{hit['text']}")
    if snapshot.next == ("review",):
        st.subheader("Human review required")
        st.text(result.get("draft_response", ""))
        action = st.selectbox("Decision", ["approve", "edit", "reject"])
        reviewer = st.text_input("Reviewer")
        reason = st.text_input("Decision reason")
        response = st.text_area("Edited customer response")
        if st.button("Submit decision"):
            try:
                resume_review(
                    graph, config, dict(action=action, reviewer=reviewer, reason=reason, response=response)
                )
                st.rerun()
            except ValueError:
                st.warning(
                    "Supply reviewer and reason; edits need a response. Approval needs a grounded draft."
                )
            except Exception as exc:
                st.error(safe_failure(exc))
    st.subheader("Final response")
    st.text(result.get("final_response", ""))

    if not snapshot.next and result.get("grounding_status") == "passed" and not result.get("human_decision"):
        if st.button("Request additional human review"):
            try:
                request_review(graph, config)
                st.rerun()
            except Exception as exc:
                st.error(safe_failure(exc))
