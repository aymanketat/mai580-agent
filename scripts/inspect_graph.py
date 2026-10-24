"""Print the graph as Mermaid and stream one run node by node.

    python scripts/inspect_graph.py                    # diagram only
    python scripts/inspect_graph.py "your question"    # diagram + streamed run

The streamed run uses the same step budget and fallback message as run.py: when the agent loop
reaches RECURSION_LIMIT, the run stops and the fallback message is shown instead of a traceback.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # author names are not cp1252

from langchain_core.messages import HumanMessage          # noqa: E402
from langgraph.errors import GraphRecursionError          # noqa: E402
from app.graph import RECURSION_LIMIT, STEP_BUDGET_MESSAGE, build_graph   # noqa: E402


def stream_run(graph, question: str, recursion_limit: int = RECURSION_LIMIT, out=print) -> dict:
    """Stream one run, printing each node's update. Returns what happened, for tests and scripts."""
    steps, final, budget_exceeded = 0, None, False
    t0 = time.perf_counter()
    try:
        for update in graph.stream({"messages": [HumanMessage(question)]}, stream_mode="updates",
                                   config={"recursion_limit": recursion_limit}):
            steps += 1
            node, payload = next(iter(update.items()))
            out(f"\n--- {node} ---")
            for key, value in (payload or {}).items():
                if key == "messages":
                    for m in value:
                        usage = getattr(m, "usage_metadata", None)
                        out(f"[{m.type}] {str(m.content)[:600]}")
                        if usage:
                            out(f"    tokens: {usage}")
                        if m.type == "ai" and m.content:
                            final = m.content
                else:
                    out(f"{key}: {str(value)[:600]}")
    except GraphRecursionError:                           # Stage 5: the step budget ran out
        budget_exceeded, final = True, STEP_BUDGET_MESSAGE
        out(f"\n--- step budget of {recursion_limit} reached ---\n{STEP_BUDGET_MESSAGE}")
    out(f"\nlatency: {time.perf_counter() - t0:.2f}s, steps: {steps}")
    return {"steps": steps, "budget_exceeded": budget_exceeded, "answer": final}


if __name__ == "__main__":
    graph = build_graph()
    print(graph.get_graph().draw_mermaid())
    if len(sys.argv) > 1:
        stream_run(graph, " ".join(sys.argv[1:]))
