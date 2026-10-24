"""Compare two result files from run_questions.py on the same question set.

    python scripts/compare_results.py eval/results/stage01_on_stage02set.jsonl eval/results/stage02.jsonl

Prints a Markdown table of latency and tokens per question, the totals, and then each pair of
answers in full so they can be graded side by side.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


a, b = load(sys.argv[1]), load(sys.argv[2])
name_a, name_b = Path(sys.argv[1]).stem, Path(sys.argv[2]).stem
assert [r["text"] for r in a] == [r["text"] for r in b], "the two files must cover the same questions"

print(f"| # | label | {name_a} s | {name_b} s | {name_a} tokens in/out | {name_b} tokens in/out |")
print("|---|---|---|---|---|---|")
for i, (ra, rb) in enumerate(zip(a, b), 1):
    print(f"| {i} | {ra.get('label', '')} | {ra['latency_s']} | {rb['latency_s']} | "
          f"{ra['tokens']['input']}/{ra['tokens']['output']} | {rb['tokens']['input']}/{rb['tokens']['output']} |")
tot = lambda rows, k: sum(r["tokens"][k] for r in rows)
print(f"| **total** | | {sum(r['latency_s'] for r in a):.1f} | {sum(r['latency_s'] for r in b):.1f} | "
      f"{tot(a, 'input')}/{tot(a, 'output')} | {tot(b, 'input')}/{tot(b, 'output')} |")

for i, (ra, rb) in enumerate(zip(a, b), 1):
    print(f"\n{'=' * 100}\n[{i}] ({ra.get('label', '')}) {ra['text']}")
    print(f"\n--- {name_a} ({len((ra['answer'] or '').split())} words)\n{ra['answer']}")
    print(f"\n--- {name_b} ({len((rb['answer'] or '').split())} words)\n{rb['answer']}")
