# Acceptance report — 2026-10-03

READY FOR PR — local workflow implementation and offline verification.
This does not certify production performance or live-model quality.

Modified: app.py, agents.py, graph.py, kb.py, llm_config.py, embedding_config.py,
requirements.txt, .env.example, .gitignore, README.md. The policy corpus is unchanged.
Added: policy.py, runtime.py, AUDIT.md, requirements-dev.txt, constraints.txt,
pyproject.toml, tests/, evaluation/, .github/workflows/ci.yml, verification.json,
and this report. All files appear as additions in Git because the imported project
had no repository or baseline commit.

Architecture: retained Streamlit + LangGraph + Chroma; replaced MemorySaver with
SQLite and structured interrupts; deterministic queue/SLA rules, evidence topic
and relevance gates, source-ID/fact checks, and extractive specialist drafts.
Compose is deterministic so it cannot add an unvalidated policy fact.

Fixed: embedding selection propagation; session-owned credentials; immutable trace
updates; sanitized UI failures; changed-KB index invalidation; lost evidence IDs;
unsupported auto-answering; volatile review state; empty/invalid decisions; unsafe
trace HTML; pending-review discovery API mismatch; stale-policy approval.

Verification: 51 pytest tests pass (3 upstream Chroma fixture deprecation warnings).
Ruff, compile/import, pip check and offline constrained-install dry run pass.
Secret-pattern scan found no provider keys, AWS access keys, GitHub tokens or private
keys in the deliverable source. Credentials, SQLite data, raw third-party datasets
and generated customer-text samples are verified ignored. This is a bounded pattern
scan, not a full external dependency security audit.

Executed evaluation: 18 native acceptance cases, 32 Bitext, 32 CLINC OOS and 8 reviewed
SaaS initial-turn cases. All used fake intake and deterministic lexical retrieval;
actual Chroma integration separately runs with fixed fake embeddings in pytest.
Native category/priority accuracy and macro-F1 are 1.0; review precision/recall/F1
are 1.0; Hit@1/Hit@2/Recall@2 and evidence-supported response rate are 1.0; false-safe
and grounding violations are zero; review rate 44.44%, automatic-draft coverage
55.56%, assisted coverage 0% (reviews remain pending during eval).
Native median/p95 latency and call counts are recorded in evaluation/results.json.
External category accuracy: Bitext 37.5%, CLINC 90.625%, SaaS 25%. False-safe is zero
across final external runs. These weak fixture routing scores remain visible and
must not be advertised as live-model metrics. SaaS priority accuracy is 12.5%.

GitHub Actions is added but unexecuted remotely: no remote was supplied. The local
equivalent checks pass. Python 3.12 constraints were verified on Windows; Ubuntu
resolution is left to the first hosted run. No paid calls, push, PR or merge occurred.

Limitations: six-case lexical calibration; extractive responses; no live-model quality
measurement; no production authentication, verified reviewer identity, tenant isolation,
retention policy or multi-operator concurrency guarantee. Human edits are not machine
grounded. Sending replies and performing refunds/account actions remain human-controlled.
Optional Twitter/BANKING77 datasets were not used. Code license remains an owner choice.

Git: feature/evidence-grounded-triage; no prior HEAD; verified files prepared in the
index for review; no commit created and no remote configured.
