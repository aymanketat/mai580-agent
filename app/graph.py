"""build_graph(): all the wiring lives here. You write this file."""
from langgraph.graph import StateGraph, START, END

from app.nodes import (
    analyze, generate, review,
    classify, route_by_intent,
    answer_paper, answer_search, answer_citation, answer_concept, answer_general,
)
from app.state import State

# Stage 3: every intent maps to exactly one answer node.
ROUTES = {
    "paper_lookup": "answer_paper",
    "literature_search": "answer_search",
    "citation": "answer_citation",
    "concept": "answer_concept",
    "other": "answer_general",
}


def build_workflow():
    """Stage 2: the fixed path analyze -> generate -> review, kept as a graph of its own."""
    builder = StateGraph(State)
    builder.add_node("analyze", analyze)
    builder.add_node("generate", generate)
    builder.add_node("review", review)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "generate")
    builder.add_edge("generate", "review")
    builder.add_edge("review", END)
    return builder.compile()


def build_graph():
    # Stage 3: classify, then a conditional edge picks the answer node from the intent.
    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_node("answer_paper", answer_paper)
    builder.add_node("answer_search", answer_search)
    builder.add_node("answer_citation", answer_citation)
    builder.add_node("answer_concept", answer_concept)
    builder.add_node("answer_general", answer_general)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_by_intent, ROUTES)
    for answer_node in set(ROUTES.values()):
        builder.add_edge(answer_node, END)
    return builder.compile()
