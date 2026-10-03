# Verified baseline audit — 2026-10-03

Supplied path: `support_triage/data`. Actual project root: `support_triage`.
At initial inspection, the imported local folder lacked `.git`; no local Git
repository was found in the root or its parents. It was subsequently reattached
to authoritative GitHub `main @ 157fa6f`, restoring the baseline/history context.
Original authoritative baseline: `157fa6f27897772ededacef3623ad693067bb294`.
Remote: `https://github.com/mimohnaik-git/FDE_Project_Support-Triage.git`.
Feature commit: `0b33ad8943cd05e370be1a36b16156f3d5ba4f5f`.
PR [#1](https://github.com/mimohnaik-git/FDE_Project_Support-Triage/pull/1) merged successfully.
Current GitHub `main`: `a41c0e141a5885483c950426a5d3d5cef6671198`.
Post-merge Ubuntu GitHub Actions run [37110362584](https://github.com/mimohnaik-git/FDE_Project_Support-Triage/actions/runs/37110362584):
**SUCCESS**, with 51 tests passed.
Final recommendation: **ACCEPTANCE PASS — PORTFOLIO READY**.
Existing evaluation metrics and limitations remain unchanged; live-model quality
remains unmeasured. This documentation reconciliation makes no commit or push.

Original files: `.env.example`, `.gitignore`, `agents.py`, `app.py`,
`embedding_config.py`, `graph.py`, `kb.py`, `llm_config.py`, `README.md`,
`requirements.txt`, and `data/kb_articles.json`. The core modules have since
moved under `support_triage/`.

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
