# Multi-Agent Customer Support Triage (LangGraph + Streamlit)

A LangGraph pipeline that triages incoming customer support tickets:

```
              ┌────────────────┐
              │  intake_agent   │   classifies category + priority
              └───────┬─────────┘
                       │ (conditional routing)
        ┌──────────────┼──────────────┬───────────────┐
        ▼              ▼              ▼                ▼
  billing_agent  technical_agent  general_agent   human_review_node
        │              │              │           (graph PAUSES here —
        │              │              │            waits for a human
        └──────┬───────┴──────┬───────┘             supervisor's notes)
               ▼               ▼
               compose_agent ◄─┘
                   │
                  END
```

Specialist agents pull grounding context from a Chroma vector DB seeded
with sample knowledge-base articles. Escalation tickets pause the graph
(`interrupt_before=["human_review_node"]`) until a human supervisor adds
notes in the UI, then resume.

Runs on **OpenAI, Anthropic, or a free local Ollama model** for the chat
LLM, and on **OpenAI or a free local embedding model** for the knowledge
base — pick whichever you have in the sidebar, or leave both on **Auto**
and it uses whatever's available. This means the demo works in front of a
client with zero API keys: switch both to their free/local option and it
runs entirely offline (after a one-time local model download).

```
support_triage/
├── app.py                # Streamlit UI (ticket input, live trace, provider picker)
├── graph.py               # LangGraph StateGraph wiring
├── agents.py              # State schema + all agent node functions
├── llm_config.py           # Picks OpenAI / Anthropic / Ollama for the chat LLM
├── embedding_config.py      # Picks OpenAI / local (free) for KB embeddings
├── kb.py                    # Chroma DB setup + retrieval
├── data/
│   └── kb_articles.json     # Seed knowledge-base content
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

> `chroma_db/` is not committed to the repo — it's a local vector-store
> cache that `kb.py` builds automatically from `data/kb_articles.json` the
> first time you run the app (or `python kb.py`). This keeps the repo free
> of binary blobs and means the KB always matches what's in the JSON file.
> Each embedding provider gets its own collection, so switching providers
> never mixes incompatible vectors.

## 1. Setup (in VS Code)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste your OPENAI_API_KEY or ANTHROPIC_API_KEY
# — or skip this entirely and use the free Ollama + local-embeddings path
```

## 2. Pick your providers — pick whichever you have

**Chat LLM** (drives all the agent reasoning):

| Provider | Cost | Setup |
|---|---|---|
| **OpenAI** | Paid | Set `OPENAI_API_KEY`. Default model: `gpt-4o-mini`. |
| **Anthropic** | Paid | Set `ANTHROPIC_API_KEY`. Default model: `claude-haiku-4-5`. |
| **Ollama** | Free, local | Install [Ollama](https://ollama.com), `ollama serve`, `ollama pull llama3.1`. No key needed. |

**Knowledge-base embeddings** (drives what context the specialists retrieve):

| Provider | Cost | Setup |
|---|---|---|
| **OpenAI** | Paid | `text-embedding-3-small`. Uses the same `OPENAI_API_KEY`. |
| **Local** | Free | `all-MiniLM-L6-v2`, runs on-device via chromadb's bundled ONNX runtime. ~80MB one-time download, then fully offline. |

The sidebar shows live ✅ / ⚠️ status for every chat provider so it's
obvious what's usable before you run a ticket. Set
`LLM_PROVIDER=openai|anthropic|ollama` and/or
`EMBEDDING_PROVIDER=openai|local` in `.env` to pin a choice instead of
auto-detecting.

## 3. Run the app

```bash
streamlit run app.py
```

Opens `http://localhost:8501`.

## 4. Demo it in class / to a client

1. Check the sidebar — confirm which chat + embedding providers are
   active (or add a key / switch to the free options right there).
2. Click a **sample ticket** in the sidebar (billing, technical, general,
   or escalation) — or paste your own.
3. Click **🚀 Run triage** and watch the **agent trace** render as a live
   timeline: intake classifies it, a specialist drafts a response using
   retrieved KB articles, and compose finalizes it.
4. Expand **📚 Knowledge base articles used** to show exactly what
   grounded the response — useful for a "why did it say that" conversation.
5. Try the **escalation sample** to show the human-in-the-loop pause: the
   graph stops before `human_review_node`, and you add supervisor notes
   in the UI to resume it and get the final response.
6. Click **🔄 Reset conversation** to start clean for the next demo.

## How the human-in-the-loop pause works

`graph.py` compiles the `StateGraph` with
`interrupt_before=["human_review_node"]`. When a ticket is classified as
`escalation`, `graph_app.invoke(...)` returns as soon as it reaches that
node without running it. `app.py` checks `graph_app.get_state(config).next`
to detect the pause, shows a notes box, then resumes with
`graph_app.update_state(...)` followed by `graph_app.invoke(None, ...)` —
passing `None` tells LangGraph to continue from where it left off using the
`MemorySaver` checkpoint for that `thread_id`.

**Note:** `MemorySaver` keeps checkpoints in memory only — restarting the
Streamlit process mid-escalation-review loses that paused thread. Fine for
a demo; for production, swap in `SqliteSaver` or `PostgresSaver`.

## How provider switching works

`llm_config.py` and `embedding_config.py` each resolve a provider once per
selection and are read by `agents.py` / `kb.py` through simple factory
functions (`build_agents()`, `get_collection()`) — no other code needs to
know which backend is actually running. Changing either dropdown in the
sidebar rebuilds the graph with the new backend on the next run.

## Extending the demo

- **Swap MemorySaver for persistence**: use `langgraph.checkpoint.sqlite.SqliteSaver`
  so paused escalations survive a restart.
- **Add more KB articles**: edit `data/kb_articles.json`, then run
  `python kb.py` or call `kb.reset_kb()` to re-seed.
- **Add more providers**: `llm_config.py` and `embedding_config.py` are
  intentionally small — add a branch and a default model entry for any
  other LangChain-supported provider (Azure, Bedrock, Gemini, etc.).
