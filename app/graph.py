"""build_graph(): all the wiring lives here. You write this file."""
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.nodes import (
    analyze, generate, review,
    classify, route_by_intent,
    answer_concept, answer_general,
    agent, final_answer,
)
from app.state import State
from app.tools import TOOLS

# Stage 3 mapped every intent to its own answer node. Stage 4 sends the three intents that
# need paper data into one tool-using branch; concept and other are still answered directly.
ROUTES = {
    "paper_lookup": "agent",
    "literature_search": "agent",
    "citation": "agent",
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
    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_node("agent", agent)
    # handle_tool_errors=True turns any exception inside a tool into an error ToolMessage
    # instead of aborting the run (the default only does that for bad arguments).
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    builder.add_node("final_answer", final_answer)
    builder.add_node("answer_concept", answer_concept)
    builder.add_node("answer_general", answer_general)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_by_intent, ROUTES)
    # Stage 4: ONE round of tools. tools_condition -> "tools" if the last message has
    # tool_calls, else END. After the tools run, final_answer writes the reply; the model
    # never gets to look at a result and call another tool.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "final_answer")
    builder.add_edge("final_answer", END)
    builder.add_edge("answer_concept", END)
    builder.add_edge("answer_general", END)
    return builder.compile()
