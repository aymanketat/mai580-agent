from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from app.graph import RECURSION_LIMIT, STEP_BUDGET_MESSAGE, build_graph

graph = build_graph()

while True:
    question = input("you> ").strip()
    if question in {"exit", "quit"}:
        break
    try:
        result = graph.invoke({"messages": [HumanMessage(question)]}, {"recursion_limit": RECURSION_LIMIT})
    except GraphRecursionError:                           # Stage 5: the step budget ran out
        result = {"messages": [AIMessage(STEP_BUDGET_MESSAGE)]}
    print("agent>", result["messages"][-1].content)
