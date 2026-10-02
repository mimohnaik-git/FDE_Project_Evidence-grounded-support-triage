"""
app.py
Streamlit front end for the LangGraph Customer Support Triage demo.

Run with: streamlit run app.py
"""

import html
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from embedding_config import DEFAULT_MODELS as EMBED_DEFAULT_MODELS
from embedding_config import resolve_embedding_provider
from llm_config import DEFAULT_MODELS as LLM_DEFAULT_MODELS
from llm_config import check_providers, resolve_provider

load_dotenv()

st.set_page_config(page_title="Multi-Agent Support Triage", page_icon="🎫", layout="wide")

## Custom CSS
# NOTE: CSS only supports /* ... */ comments, not "# ...". Keep using /* */
# here — a "#" comment merges into the *selector* of the next rule and
# silently breaks it (e.g. "# status badges\n.badge" parses as the single,
# never-matching selector "#status badges .badge").
st.markdown("""
<style>
    /* Hide streamlit toolbar */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        opacity: 0.75;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-billing    { background: #dbeafe; color: #1e40af; }
    .badge-technical  { background: #fef3c7; color: #92400e; }
    .badge-general    { background: #e0e7ff; color: #3730a3; }
    .badge-escalation { background: #fee2e2; color: #991b1b; }
    .badge-high       { background: #fee2e2; color: #991b1b; }
    .badge-medium     { background: #fef9c3; color: #854d0e; }
    .badge-low        { background: #dcfce7; color: #166534; }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        .badge-billing    { background: #1e3a8a; color: #93c5fd; }
        .badge-technical  { background: #78350f; color: #fcd34d; }
        .badge-general    { background: #312e81; color: #a5b4fc; }
        .badge-escalation { background: #7f1d1d; color: #fca5a5; }
        .badge-high       { background: #7f1d1d; color: #fca5a5; }
        .badge-medium     { background: #713f12; color: #fde68a; }
        .badge-low        { background: #14532d; color: #86efac; }
    }

    /* Timeline */
    .step {
        display: flex;
        gap: 12px;
        padding: 10px 0;
        border-left: 3px solid rgba(128, 128, 128, 0.3);
        margin-left: 8px;
        padding-left: 16px;
        position: relative;
    }
    .step:last-child { border-left-color: transparent; }
    .step::before {
        content: "";
        position: absolute;
        left: -7px;
        top: 14px;
        width: 11px;
        height: 11px;
        border-radius: 50%;
        background: #94a3b8;
        border: 2px solid var(--background-color, #ffffff);
        box-shadow: 0 0 0 1px rgba(128, 128, 128, 0.4);
    }
    .step.done::before { background: #22c55e; box-shadow: 0 0 0 1px #86efac; }
    .step.active::before { background: #3b82f6; box-shadow: 0 0 0 1px #93c5fd; }

    /* Final response box */
    .response-box {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        font-size: 0.95rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }

    /* Human review panel */
    .human-panel {
        background: rgba(249, 115, 22, 0.08);
        border: 1px solid rgba(249, 115, 22, 0.3);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 8px;
        text-align: left;
        justify-content: flex-start;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: model provider picker
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Model")

PROVIDER_LABELS = {
    "auto": "Auto (use whichever is available)",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "ollama": "Ollama (free, local)",
}
llm_provider_choice = st.sidebar.selectbox(
    "LLM provider",
    options=list(PROVIDER_LABELS.keys()),
    format_func=lambda p: PROVIDER_LABELS[p],
    index=0,
    help="Auto picks the first available option: OpenAI key -> Anthropic key -> local Ollama.",
)

statuses = {s.provider: s for s in check_providers()}
icon = {"openai": "🟢", "anthropic": "🟣", "ollama": "🖥️"}
for name in ("openai", "anthropic", "ollama"):
    s = statuses[name]
    mark = "✅" if s.available else "⚠️"
    st.sidebar.caption(f"{icon[name]} **{PROVIDER_LABELS[name]}** {mark} — {s.detail}")

with st.sidebar.expander("Add / change API keys"):
    openai_key_input = st.text_input(
        "OpenAI API key", value=os.environ.get("OPENAI_API_KEY", ""), type="password",
        help="Also used for KB embeddings if the embedding provider below is set to OpenAI.",
    )
    if openai_key_input:
        os.environ["OPENAI_API_KEY"] = openai_key_input

    anthropic_key_input = st.text_input(
        "Anthropic API key", value=os.environ.get("ANTHROPIC_API_KEY", ""), type="password"
    )
    if anthropic_key_input:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key_input

    st.caption(
        "No key handy? Install [Ollama](https://ollama.com), run "
        "`ollama pull llama3.1`, then pick **Ollama (free, local)** above."
    )

EMBED_LABELS = {"auto": "Auto (OpenAI if a key is set, else free local)", "openai": "OpenAI (paid)", "local": "Local — MiniLM (free)"}
embed_provider_choice = st.sidebar.selectbox(
    "Knowledge-base embeddings",
    options=list(EMBED_LABELS.keys()),
    format_func=lambda p: EMBED_LABELS[p],
    index=0,
    help="Local embeddings run fully on-device (no key, no cost) after a one-time model download.",
)

active_llm_provider = resolve_provider(llm_provider_choice)
active_embed_provider = resolve_embedding_provider(embed_provider_choice)
st.sidebar.caption(
    f"Active: **{active_llm_provider}** / `{LLM_DEFAULT_MODELS[active_llm_provider]}` chat model, "
    f"**{active_embed_provider}** / `{EMBED_DEFAULT_MODELS[active_embed_provider]}` embeddings"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Try a sample ticket:**")

SAMPLES = {
    "Billing — double charge": "I was charged twice for my monthly subscription this "
    "cycle. Can you check and refund the extra charge?",
    "Technical — can't log in": "I keep getting 'invalid credentials' when I try to log "
    "in even though I'm sure my password is correct. I've tried resetting it too.",
    "General — support hours": "What are your support hours and how do I reach someone "
    "on the weekend?",
    "Escalation — angry customer": "This is the THIRD time I'm contacting you. Nobody "
    "has fixed my billing issue in 2 weeks and I'm being charged for a service I "
    "cancelled. Fix this today or I'm disputing the charge with my bank and posting "
    "about this everywhere.",
}
for label, text in SAMPLES.items():
    if st.sidebar.button(label, use_container_width=True):
        st.session_state["ticket_text"] = text

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset conversation"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🎫 Multi-Agent Customer Support Triage")
st.caption(
    "LangGraph orchestrates Intake → (Billing / Technical / General / Human-Escalation) "
    "→ Compose. Specialist agents pull context from a Chroma vector DB."
)

if not any(s.available for s in statuses.values()):
    st.warning(
        "No LLM provider is available yet. Add an OpenAI or Anthropic key above, "
        "or install [Ollama](https://ollama.com) and run `ollama pull llama3.1`."
    )
    st.stop()

# Build (and cache) the graph app; rebuild if the provider selection changes.
provider_key = (llm_provider_choice, embed_provider_choice)
if st.session_state.get("provider_key") != provider_key or "graph_app" not in st.session_state:
    from graph import build_graph  # imported here so provider keys are set before any client init

    try:
        graph_app, resolved_llm_provider, resolved_llm_model = build_graph(provider=llm_provider_choice)
    except RuntimeError as exc:
        st.error(f"Couldn't set up the model: {exc}")
        st.stop()

    st.session_state["graph_app"] = graph_app
    st.session_state["resolved_llm"] = (resolved_llm_provider, resolved_llm_model)
    st.session_state["provider_key"] = provider_key
    # A new backend means old thread state no longer applies.
    st.session_state.pop("last_result", None)
    st.session_state.pop("awaiting_human", None)

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

graph_app = st.session_state["graph_app"]
config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

ticket_text = st.text_area(
    "Customer ticket",
    value=st.session_state.get("ticket_text", ""),
    height=120,
    placeholder="Paste or type the customer's message here...",
    key="ticket_text",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🚀 Run triage", type="primary", use_container_width=True)

if submit and ticket_text.strip():
    st.session_state["thread_id"] = str(uuid.uuid4())  # fresh thread per new ticket
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    initial_state = {
        "ticket_id": st.session_state["thread_id"][:8],
        "customer_message": ticket_text,
        "trace": [],
        "kb_hits": [],
        "needs_human": False,
        "human_notes": None,
    }

    try:
        with st.spinner("Agents are working..."):
            result = graph_app.invoke(initial_state, config=config)
        st.session_state["last_result"] = result
        st.session_state["awaiting_human"] = bool(graph_app.get_state(config).next)
        st.session_state["run_error"] = None
    except Exception as exc:
        st.session_state["last_result"] = None
        st.session_state["run_error"] = str(exc)

# ---------------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------------
if st.session_state.get("run_error"):
    st.error(f"Triage run failed: {st.session_state['run_error']}")
    if "OPENAI_API_KEY" in st.session_state["run_error"] or "ANTHROPIC_API_KEY" in st.session_state["run_error"]:
        st.info("Add a key in the sidebar, or switch the LLM provider to Ollama.")
    elif "onnx" in st.session_state["run_error"].lower() or "s3.amazonaws" in st.session_state["run_error"]:
        st.info("The free local embedding model needs one-time internet access to download. "
                "If this network blocks it, switch the embeddings provider to OpenAI in the sidebar.")

result = st.session_state.get("last_result")
resolved_llm_provider, resolved_llm_model = st.session_state.get("resolved_llm", (None, None))

if result:
    if resolved_llm_provider:
        st.caption(f"This run used **{resolved_llm_provider}** / `{resolved_llm_model}`")

    category = result.get("category", "—")
    priority = result.get("priority", "—")

    badge_cols = st.columns(3)
    badge_cols[0].markdown(
        f"**Category**<br><span class='badge badge-{category}'>{category}</span>",
        unsafe_allow_html=True,
    )
    badge_cols[1].markdown(
        f"**Priority**<br><span class='badge badge-{priority}'>{priority}</span>",
        unsafe_allow_html=True,
    )
    badge_cols[2].metric("Ticket ID", result.get("ticket_id", "—"))

    if result.get("reasoning"):
        st.caption(f"🧠 Routing reasoning: {result['reasoning']}")

    st.markdown("#### 🧭 Agent trace")
    trace = result.get("trace", [])
    for i, step in enumerate(trace):
        step_class = "done" if i < len(trace) - 1 or not st.session_state.get("awaiting_human") else "active"
        st.markdown(f"<div class='step {step_class}'>{step}</div>", unsafe_allow_html=True)

    if result.get("kb_hits"):
        with st.expander("📚 Knowledge base articles used"):
            for hit in result["kb_hits"]:
                st.markdown(f"**{hit['title']}** — {hit['text']}")

    st.markdown("---")

    # -----------------------------------------------------------------
    # Human-in-the-loop panel (escalation path only)
    # -----------------------------------------------------------------
    if st.session_state.get("awaiting_human"):
        st.markdown("### 🧑‍💼 Human review required")
        st.markdown(
            "<div class='human-panel'>This ticket was routed to <b>escalation</b>. "
            "LangGraph has paused the graph before the human_review_node. Add "
            "supervisor notes below to resume it.</div>",
            unsafe_allow_html=True,
        )
        notes = st.text_area(
            "Supervisor notes (what should the agent tell the customer?)",
            placeholder="e.g. Approve full refund, apologize for the delay, escalate "
            "internally to billing ops.",
            key="human_notes_input",
        )
        if st.button("▶️ Resume graph with these notes"):
            try:
                with st.spinner("Resuming graph..."):
                    graph_app.update_state(config, {"human_notes": notes})
                    resumed = graph_app.invoke(None, config=config)
                st.session_state["last_result"] = resumed
                st.session_state["awaiting_human"] = bool(graph_app.get_state(config).next)
                st.session_state["run_error"] = None
            except Exception as exc:
                st.session_state["run_error"] = str(exc)
            st.rerun()
    else:
        if result.get("final_response"):
            st.markdown("### ✉️ Final response to customer")
            # The response text is LLM-generated from a customer-supplied ticket,
            # so it's escaped before going into unsafe_allow_html markup — a
            # prompt-injected ticket shouldn't be able to render arbitrary HTML.
            safe_response = html.escape(result["final_response"])
            st.markdown(
                f"<div class='response-box'>{safe_response}</div>",
                unsafe_allow_html=True,
            )
elif not st.session_state.get("run_error"):
    st.info("Enter a ticket above (or pick a sample from the sidebar) and click **Run triage**.")
