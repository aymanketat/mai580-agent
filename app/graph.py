"""build_graph(): all the wiring lives here. You write this file."""
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.nodes import (
    analyze, generate, review,
    classify, route_by_intent,
    answer_concept, answer_general,
    agent,
)
from app.state import State
from app.tools import TOOLS

# Stage 5: a loop needs a stop. Each agent -> tools round is 2 steps, so 12 allows about five
# tool rounds. The installed default is 10007 (langgraph 1.2.12), not the 25 the docs mention,
# so without this an agent that keeps calling tools would make thousands of model calls.
RECURSION_LIMIT = 12
STEP_BUDGET_MESSAGE = "I could not finish this request within my step budget. Please ask a narrower question."

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
    builder.add_node("answer_concept", answer_concept)
    builder.add_node("answer_general", answer_general)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_by_intent, ROUTES)
    # The agent loop (ReAct): tools_condition -> "tools" while the last message has tool_calls,
    # else END. Stage 5 closes the loop: after the tools run, control returns to the agent,
    # which looks at the results and decides whether to call more tools or to answer.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")      # Stage 4 had: "tools" -> "final_answer"
    builder.add_edge("answer_concept", END)
    builder.add_edge("answer_general", END)
    return builder.compile()
