"""Execute the real graph with deterministic fake intake/retrieval and measure it."""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from evaluation.offline import FakeChat, fake_retrieve
from support_triage.graph import build_graph


def run(
    cases=None, chat=None, retriever=None, mode="offline fake intake and retrieval; workflow regression only"
):
    cases = cases or json.loads((Path(__file__).parent / "datasets" / "gold.json").read_text())
    outputs, latencies = [], []
    with tempfile.TemporaryDirectory() as folder:
        graph, _, _ = build_graph(
            chat=chat or FakeChat(),
            retriever=retriever or fake_retrieve,
            checkpoint_path=Path(folder) / "eval.sqlite",
        )
        for case in cases:
            start = time.perf_counter()
            output = graph.invoke(
                {"ticket_id": case["id"], "customer_message": case["message"], "trace": []},
                {"configurable": {"thread_id": case["id"]}},
            )
            latencies.append((time.perf_counter() - start) * 1000)
            outputs.append(output)
        graph.checkpointer.conn.close()
    metrics = {"mode": mode, "cases": len(cases)}
    for field in ("category", "priority"):
        labeled = [(c[field], o[field]) for c, o in zip(cases, outputs) if c.get(field) is not None]
        metrics[f"{field}_labeled_cases"] = len(labeled)
        if labeled:
            expected, actual = zip(*labeled)
            metrics[f"{field}_accuracy"] = accuracy_score(expected, actual)
            metrics[f"{field}_macro_f1"] = f1_score(expected, actual, average="macro", zero_division=0)
        else:
            metrics[f"{field}_accuracy"] = None
            metrics[f"{field}_macro_f1"] = None
    expected = [c["review"] for c in cases]
    actual = [o["human_review_required"] for o in outputs]
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, actual, average="binary", zero_division=0
    )
    metrics.update(
        escalation_precision=precision,
        escalation_recall=recall,
        escalation_f1=f1,
        false_safe_rate=sum(e and not a for e, a in zip(expected, actual)) / max(1, sum(expected)),
        human_review_rate=statistics.mean(actual),
        automation_coverage=statistics.mean(not a for a in actual),
        assisted_handling_coverage=statistics.mean(
            bool(o.get("human_decision") and o.get("final_response")) for o in outputs
        ),
    )
    oos = [(c, o) for c, o in zip(cases, outputs) if c["category"] == "unsupported"]
    metrics["oos_abstention_accuracy"] = (
        statistics.mean(o["human_review_required"] for c, o in oos) if oos else None
    )
    retrieval = [(c, o) for c, o in zip(cases, outputs) if c["relevant_ids"]]
    for k in (1, 2):
        metrics[f"hit_at_{k}"] = (
            statistics.mean(
                bool(set(c["relevant_ids"]) & {h["id"] for h in o["kb_hits"][:k]}) for c, o in retrieval
            )
            if retrieval
            else None
        )
    metrics["recall_at_2"] = (
        statistics.mean(
            len(set(c["relevant_ids"]) & {h["id"] for h in o["kb_hits"][:2]}) / len(c["relevant_ids"])
            for c, o in retrieval
        )
        if retrieval
        else None
    )
    answers = [o for o in outputs if o.get("final_response")]
    metrics["evidence_supported_response_rate"] = (
        statistics.mean(o["grounding_status"] == "passed" for o in answers) if answers else None
    )
    metrics["grounding_violations"] = sum(
        o.get("final_response") != "" and o["grounding_status"] != "passed" for o in outputs
    )
    metrics["latency_median_ms"] = statistics.median(latencies)
    metrics["latency_p95_ms"] = sorted(latencies)[max(0, int(len(latencies) * 0.95 + 0.999) - 1)]
    metrics["llm_calls_per_run"] = statistics.mean(o["llm_calls"] for o in outputs)
    metrics["retrieval_calls_per_run"] = statistics.mean(o["retrieval_calls"] for o in outputs)
    metrics["case_results"] = [
        dict(id=c["id"], category=o["category"], review=o["human_review_required"])
        for c, o in zip(cases, outputs)
    ]
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/native.json"))
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--live", action="store_true", help="Explicit opt-in: may use paid APIs")
    parser.add_argument("--provider", choices=["auto", "openai", "anthropic", "ollama"], default="auto")
    parser.add_argument("--embedding-provider", choices=["auto", "local", "openai"], default="local")
    parser.add_argument("--model")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text()) if args.cases else None
    if args.live:
        from support_triage import kb
        from support_triage.llm_config import get_chat_model
        from support_triage.runtime import Credentials

        credentials = Credentials.from_environment()
        chat, provider, model = get_chat_model(args.provider, args.model, credentials=credentials)

        def retriever(query, **kwargs):
            return kb.retrieve(
                query,
                category=kwargs.get("category"),
                k=kwargs.get("k", 2),
                provider=args.embedding_provider,
                credentials=credentials,
            )

        result = run(
            cases,
            chat,
            retriever,
            mode=f"live {provider}/{model}; embedding selection {args.embedding_provider}",
        )
    else:
        result = run(cases)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "case_results"}, indent=2))
    if result["false_safe_rate"] or result["grounding_violations"]:
        raise SystemExit("Safety regression")
