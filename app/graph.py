"""build_graph(): all the wiring lives here. You write this file."""
from langgraph.graph import StateGraph, START, END

from app.nodes import analyze, generate, review
from app.state import State


def build_graph():
    # Stage 2: a fixed path. START -> analyze -> generate -> review -> END.
    # Nodes share data through state fields (analysis, draft), not return values.
    builder = StateGraph(State)
    builder.add_node("analyze", analyze)
    builder.add_node("generate", generate)
    builder.add_node("review", review)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "generate")
    builder.add_edge("generate", "review")
    builder.add_edge("review", END)
    return builder.compile()
