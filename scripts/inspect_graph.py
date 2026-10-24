"""Print the graph as Mermaid and stream one run node by node.

    python scripts/inspect_graph.py                    # diagram only
    python scripts/inspect_graph.py "your question"    # diagram + streamed run
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # author names are not cp1252

from langchain_core.messages import HumanMessage          # noqa: E402
from app.graph import build_graph                         # noqa: E402

graph = build_graph()
print(graph.get_graph().draw_mermaid())

if len(sys.argv) > 1:
    question = " ".join(sys.argv[1:])
    t0 = time.perf_counter()
    for update in graph.stream({"messages": [HumanMessage(question)]}, stream_mode="updates"):
        node, payload = next(iter(update.items()))
        print(f"\n--- {node} ---")
        for key, value in payload.items():
            if key == "messages":
                for m in value:
                    usage = getattr(m, "usage_metadata", None)
                    print(f"[{m.type}] {m.content[:600]}")
                    if usage:
                        print(f"    tokens: {usage}")
            else:
                print(f"{key}: {str(value)[:600]}")
    print(f"\nlatency: {time.perf_counter() - t0:.2f}s")
