"""Stage 5: build the same loop with langchain's create_agent and print both diagrams.

    python scripts/compare_prebuilt.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.agents import create_agent                 # noqa: E402
from app.graph import build_graph                         # noqa: E402
from app.llm import model                                 # noqa: E402
from app.tools import TOOLS                               # noqa: E402

print("=== hand-built loop (app/graph.py)")
print(build_graph().get_graph().draw_mermaid())

prebuilt = create_agent(model, tools=TOOLS,
                        system_prompt="You are a research assistant for machine-learning papers.")
print("=== create_agent(model, tools=TOOLS)")
print(prebuilt.get_graph().draw_mermaid())
