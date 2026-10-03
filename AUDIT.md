# Verified baseline audit — 2026-10-03

Supplied path: `support_triage/data`. Actual project root: `support_triage`.
No Git repository existed in the root or its parents. Baseline branch: unavailable.
Baseline HEAD: unavailable. No remote or history was supplied.
Initialized `feature/evidence-grounded-triage` without an initial commit; a branch
from a prior commit is impossible for this imported source directory.

Original files: `.env.example`, `.gitignore`, `agents.py`, `app.py`,
`embedding_config.py`, `graph.py`, `kb.py`, `llm_config.py`, `README.md`,
`requirements.txt`, `data/kb_articles.json`.

Original requirements: Streamlit >=1.37,<2; LangGraph >=0.2.20,<2;
langchain-openai/anthropic/ollama >=0.2,<2; Chroma >=0.5.5,<2;
python-dotenv >=1.0.1,<2; Pydantic >=2.7,<3; httpx >=0.27,<1.
No tests, dev requirements, constraints/lock, evaluation or CI were present.

Confirmed by reading code: embedding UI selection omitted from build_graph;
UI secrets written into global os.environ; trace lists mutated in place;
MemorySaver loses pending review on restart; KB only seeded when empty;
retrieval discarded article IDs and accepted all hits; no evidence gate,
grounding validation, unsupported category or deterministic queue/SLA policy;
human notes could be empty; composer introduced an unvalidated model pass;
raw exception text and unescaped trace markup reached the UI.

Corpus: 10 authoritative policy articles. It is preserved unchanged.
Python was absent from PATH; the bundled Python 3.12.14 runtime creates the
project-local environment. No unrelated project architecture was consulted.
