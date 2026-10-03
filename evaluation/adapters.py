"""Local-file deterministic adapters. Labels are never copied into ticket input."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

BITEXT_MAP = {
    "get_refund": "billing",
    "check_refund_policy": "billing",
    "track_refund": "billing",
    "payment_issue": "billing",
    "check_payment_methods": "billing",
    "get_invoice": "billing",
    "check_invoice": "billing",
    "cancel_order": "unsupported",
    "recover_password": "technical",
    "registration_problems": "technical",
    "contact_customer_service": "general",
    "contact_human_agent": "escalation",
}


def adapt(source, rows):
    result = []
    for index, row in enumerate(rows):
        if source == "bitext":
            intent = row["intent"]
            if intent not in BITEXT_MAP:
                continue
            category, message = BITEXT_MAP[intent], row["instruction"]
        elif source == "clinc":
            message, label = row
            if label != "oos":
                continue
            category = "unsupported"
        elif source == "saas":
            # Requires independently reviewed labels; never infer escalation from agent resolutions.
            category = row["gold_category"]
            turns = row.get("messages", row.get("response", {}).get("conversation", []))
            first = next(t for t in turns if t["role"] in {"user", "customer"})
            message = first.get("content", first.get("message"))
        else:
            raise ValueError("Unsupported adapter")
        result.append(
            {
                "id": f"{source}-{index}",
                "message": message,
                "category": category,
                "priority": row.get("gold_priority") if isinstance(row, dict) else None,
                "review": row.get("gold_review", category in {"unsupported", "escalation", "ambiguous"})
                if isinstance(row, dict)
                else True,
                "relevant_ids": [],
            }
        )
    return result


def sample(rows, count=32, seed=42):
    return random.Random(seed).sample(rows, min(count, len(rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["bitext", "clinc", "saas"])
    parser.add_argument("input", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.suffix == ".csv":
        with args.input.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        content = args.input.read_text(encoding="utf-8")
        rows = (
            [json.loads(line) for line in content.splitlines()]
            if args.input.suffix == ".jsonl"
            else json.loads(content)
        )
        if args.source == "clinc":
            rows = rows["oos_test"]
    selected = sample(adapt(args.source, rows), args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    manifest = dict(
        source=args.source,
        license=args.license,
        upstream_revision=args.revision,
        sample_seed=args.seed,
        upstream_rows=len(rows),
        sample_rows=len(selected),
        purpose="Routing/OOS robustness; not company-policy acceptance",
        mapping=BITEXT_MAP if args.source == "bitext" else "OOS-only or reviewed SaaS intake labels",
        raw_data_committed=False,
        sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
    )
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
