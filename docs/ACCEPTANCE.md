# Acceptance report — 2026-10-03

ACCEPTANCE PASS — PORTFOLIO READY — verified implementation and post-merge GitHub Actions success.
This does not certify production performance or live-model quality.

Modified: app.py, support_triage/agents.py, support_triage/graph.py,
support_triage/kb.py, support_triage/llm_config.py, support_triage/embedding_config.py,
requirements.txt, .env.example, .gitignore, README.md. The policy corpus is unchanged.
Added: support_triage/policy.py, support_triage/runtime.py, docs/AUDIT.md, requirements-dev.txt, constraints.txt,
pyproject.toml, tests/, evaluation/, .github/workflows/ci.yml, verification.json,
and this report. The imported local folder initially lacked `.git`, but was
subsequently reattached to authoritative GitHub `main @ 157fa6f`.
The authoritative baseline is `157fa6f27897772ededacef3623ad693067bb294`.

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
Native median/p95 latency and call counts are recorded in evaluation/results/native.json.
External category accuracy: Bitext 37.5%, CLINC 90.625%, SaaS 25%. False-safe is zero
across final external runs. These weak fixture routing scores remain visible and
must not be advertised as live-model metrics. SaaS priority accuracy is 12.5%.

Local equivalent checks pass. Python 3.12 constraints were verified on Windows,
and the post-reorganization GitHub Actions run [37114631585](https://github.com/mimohnaik-git/evidence-grounded-support-triage/actions/runs/37114631585) completed with **SUCCESS**. PR [#1](https://github.com/mimohnaik-git/evidence-grounded-support-triage/pull/1)
merged successfully. Live-model quality remains unmeasured; no paid-call result is claimed.

Limitations: six-case lexical calibration; extractive responses; no live-model quality
measurement; no production authentication, verified reviewer identity, tenant isolation,
retention policy or multi-operator concurrency guarantee. Human edits are not machine
grounded. Sending replies and performing refunds/account actions remain human-controlled.
Optional Twitter/BANKING77 datasets were not used. Code license remains an owner choice.

GitHub remote: `https://github.com/mimohnaik-git/evidence-grounded-support-triage.git`.
Baseline commit: `157fa6f27897772ededacef3623ad693067bb294`.
Feature implementation commit: `0b33ad8943cd05e370be1a36b16156f3d5ba4f5f`.
Implementation merge: `a41c0e141a5885483c950426a5d3d5cef6671198`.
Reorganization merge: `592bafd4ea0295032310adc8ba2e99a74676f2fc`.
