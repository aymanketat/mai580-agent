"""Run a question set through the current graph, one fresh run per question.

    python scripts/run_questions.py eval/stage01_questions.jsonl eval/results/stage01.jsonl

Input: one JSON object per line with at least {"text": ...}; other keys (label, rationale)
are copied through. Output: one line per question with the answer, latency, token usage,
the node sequence, and the tool calls made. Used by the experiments in every stage.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # author names are not cp1252

from langchain_core.callbacks import UsageMetadataCallbackHandler   # noqa: E402
from langchain_core.messages import HumanMessage          # noqa: E402
from app.graph import build_graph                         # noqa: E402


def run_one(graph, text: str) -> dict:
    nodes, tool_calls = [], []
    # Tokens are counted per LLM call, not per message: nodes that store .content in a
    # state field (Stage 2) or build a fresh AIMessage carry no usage_metadata.
    usage_cb = UsageMetadataCallbackHandler()
    t0 = time.perf_counter()
    final = None
    for update in graph.stream({"messages": [HumanMessage(text)]}, stream_mode="updates",
                               config={"callbacks": [usage_cb]}):
        node, payload = next(iter(update.items()))
        nodes.append(node)
        for m in (payload or {}).get("messages", []):
            for call in getattr(m, "tool_calls", []) or []:
                tool_calls.append({"name": call["name"], "args": call["args"]})
            if m.type == "ai" and m.content:
                final = m.content
    usage = usage_cb.usage_metadata.values()           # one entry per model name
    return {
        "answer": final,
        "latency_s": round(time.perf_counter() - t0, 2),
        "tokens": {"input": sum(u["input_tokens"] for u in usage),
                   "output": sum(u["output_tokens"] for u in usage)},
        "nodes": nodes,
        "tool_calls": tool_calls,
    }


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    graph = build_graph()
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    with dst.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows, 1):
            result = {**row, **run_one(graph, row["text"])}
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] {row['text']}")
            print(f"    nodes={result['nodes']} tools={[c['name'] for c in result['tool_calls']]} "
                  f"latency={result['latency_s']}s tokens={result['tokens']}")
            print(f"    -> {(result['answer'] or '')[:300]}\n")
    print(f"wrote {len(rows)} results to {dst}")
