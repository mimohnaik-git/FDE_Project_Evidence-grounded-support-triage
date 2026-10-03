"""Select evidence threshold on calibration data separate from acceptance cases."""

import json
from pathlib import Path

from support_triage.kb import articles
from support_triage.policy import relevance


def calibrate():
    rows = json.loads((Path(__file__).parent / "datasets" / "calibration.json").read_text())
    corpus = {a["id"]: a for a in articles()}
    scores = [(relevance(r["message"], corpus[r["article_id"]]), r["sufficient"]) for r in rows]
    candidates = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]

    def cost(threshold):
        # Prefer abstention: accepting a negative costs five times a false rejection.
        return sum((5 if not label else 1) for score, label in scores if (score >= threshold) != label)

    threshold = min(candidates, key=lambda t: (cost(t), t))
    return {
        "threshold": threshold,
        "weighted_errors": cost(threshold),
        "cases": len(rows),
        "scope": "lexical evidence gate only; not embedding-distance calibration",
    }


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=2))
