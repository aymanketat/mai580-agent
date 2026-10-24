from langchain_core.messages import HumanMessage
from app.graph import build_workflow


def test_workflow_visits_three_nodes():
    graph = build_workflow()
    visited = [list(u)[0] for u in graph.stream(
        {"messages": [HumanMessage("What does LoRA stand for?")]}, stream_mode="updates")]
    assert visited == ["analyze", "generate", "review"]
