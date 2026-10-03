"""Reproduce pinned third-party samples from locally acquired raw files."""

import csv
import hashlib
import json
from pathlib import Path

from evaluation.adapters import adapt, sample
from evaluation.run import run

root = Path(__file__).resolve().parent.parent

PINNED = {
    "bitext": {
        "source": "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "revision": "430d1a89bd93bd1fa23c16f29dd53e73f0087443",
        "license": "cdla-sharing-1.0",
        "file": "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
    },
    "saas": {
        "source": "https://huggingface.co/datasets/Jurgen1161/synthetic-b2b-saas-support-dialogues-sample",
        "revision": "e3b2b9074db1ec6b1be6c7f894deb6f6588de15f",
        "license": "cc-by-4.0",
        "file": "sample.jsonl",
    },
    "clinc": {
        "source": "https://github.com/clinc/oos-eval",
        "revision": "828f8093932c8fe6ca7936c3d2e52903b1c523de",
        "license": "CC-BY-3.0 (repository LICENSE)",
        "file": "clinc.json",
    },
}


def download():
    import urllib.request

    raw = root / "evaluation" / "raw"
    raw.mkdir(exist_ok=True)
    for source, entry in PINNED.items():
        if source == "clinc":
            url = f"https://raw.githubusercontent.com/clinc/oos-eval/{entry['revision']}/data/data_full.json"
        else:
            repo = entry["source"].split("/datasets/")[1]
            url = f"https://huggingface.co/datasets/{repo}/resolve/{entry['revision']}/{entry['file']}"
        with urllib.request.urlopen(url, timeout=60) as response:
            (raw / entry["file"]).write_bytes(response.read())
    (raw / "metadata.json").write_text(json.dumps(PINNED, indent=2), encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Opt-in public dataset download")
    args = parser.parse_args()
    if args.download:
        download()
    raw = root / "evaluation" / "raw"
    generated = root / "evaluation" / "generated"
    generated.mkdir(exist_ok=True)
    metadata = json.loads((raw / "metadata.json").read_text())
    manifests = []

    for source in ("bitext", "clinc", "saas"):
        entry = metadata[source]
        source_path = raw / entry["file"]
        if source == "bitext":
            with source_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            cases = sample(adapt(source, rows), 32, 42)
            mapping = "Selected explicit Bitext intent mappings in adapters.py; ecommerce cancellation remains unsupported"
        elif source == "clinc":
            rows = json.loads(source_path.read_text())["oos_test"]
            cases = sample(adapt(source, rows), 32, 42)
            mapping = "Only oos_test utterance text; all expected unsupported/review"
        else:
            rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()]
            # Independently reviewed intake-only escalation challenge labels. Do not use resolution_status.
            labels = {
                "gen-0015": ("escalation", "high"),  # unresolved access provisioning for a week
                "gen-0312": ("escalation", "high"),  # data integrity failure under urgent migration
                "gen-0430": ("escalation", "high"),  # repeated outage with lost work
                "gen-0002": ("technical", "medium"),  # locked out of a different named product
                "gen-0038": ("technical", "medium"),  # unsupported passkey/error code scenario
                "gen-0157": ("technical", "high"),  # files missing from connected application
                "gen-0407": ("technical", "medium"),  # unsupported API 500 integration error
                "gen-0122": ("escalation", "high"),  # disclosure of billing emails to former employee
            }
            reviewed = [
                {
                    **row,
                    "gold_category": labels[row["id"]][0],
                    "gold_priority": labels[row["id"]][1],
                    "gold_review": True,
                }
                for row in rows
                if row["id"] in labels
            ]
            cases = sample(adapt(source, reviewed), 8, 42)
            mapping = "Eight manually reviewed first-customer-turn cases; labels documented in SaaS review file; all require review"
            (root / "evaluation" / "saas_labels.json").write_text(json.dumps(labels, indent=2))
        (generated / f"{source}.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
        manifest = dict(
            source=entry["source"],
            license=entry["license"],
            upstream_revision=entry["revision"],
            sample_seed=42,
            upstream_rows=len(rows),
            sample_rows=len(cases),
            purpose="Routing robustness"
            if source == "bitext"
            else "OOS abstention"
            if source == "clinc"
            else "Escalation and HITL challenge",
            mapping=mapping,
            raw_data_committed=False,
            sample_data_committed=False,
            raw_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            sample_sha256=hashlib.sha256((generated / f"{source}.json").read_bytes()).hexdigest(),
        )
        manifests.append(manifest)
        result = run(cases)
        (root / "evaluation" / f"results-{source}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(source, json.dumps({k: v for k, v in result.items() if k != "case_results"}, indent=2))

    manifests.insert(
        0,
        dict(
            source="project-native evaluation/gold.json",
            license="Project source; owner license choice pending",
            upstream_revision="gold-v1",
            sample_seed=None,
            upstream_rows=18,
            sample_rows=18,
            purpose="Authoritative workflow acceptance",
            mapping="Hand-authored intake labels and KB IDs",
            raw_data_committed=True,
        ),
    )
    (root / "evaluation" / "manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
