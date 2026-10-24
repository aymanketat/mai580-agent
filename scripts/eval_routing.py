"""Measure routing accuracy on the labelled set and print a confusion table.

    python scripts/eval_routing.py v1          # writes eval/results/routing_v1.json

Only the classify node runs (not the answer nodes), so the run is cheap. The tag names the
version of the category descriptions being measured.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage          # noqa: E402
from app.nodes import classify, Intent                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = list(Intent.__args__)

tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
rows = [json.loads(l) for l in (ROOT / "eval/routing.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

confusion = Counter()
mistakes = []
for row in rows:
    predicted = classify({"messages": [HumanMessage(row["text"])]})["intent"]
    row["predicted"] = predicted
    confusion[(row["label"], predicted)] += 1
    if predicted != row["label"]:
        mistakes.append(row)

accuracy = 1 - len(mistakes) / len(rows)
print(f"routing accuracy ({tag}): {len(rows) - len(mistakes)}/{len(rows)} = {accuracy:.0%}\n")
print("| true \\ predicted | " + " | ".join(LABELS) + " |")
print("|---|" + "---|" * len(LABELS))
for true in LABELS:
    print(f"| **{true}** | " + " | ".join(str(confusion[(true, p)] or "") for p in LABELS) + " |")
for m in mistakes:
    print(f"\nMISS  {m['label']} -> {m['predicted']}: {m['text']}")

out = ROOT / f"eval/results/routing_{tag}.json"
out.write_text(json.dumps({"tag": tag, "accuracy": accuracy, "rows": rows}, indent=1, ensure_ascii=False),
               encoding="utf-8")
print(f"\nwrote {out}")
