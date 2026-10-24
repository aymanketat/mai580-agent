"""build_graph(): all the wiring lives here. You write this file."""
from langgraph.graph import StateGraph, START, END

from app.nodes import call_model
from app.state import State


def build_graph():
    # Stage 1: one node. START -> model -> END.
    builder = StateGraph(State)
    builder.add_node("model", call_model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    return builder.compile()
