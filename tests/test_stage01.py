from langchain_core.messages import AIMessage, HumanMessage
from app.graph import build_graph


def test_graph_compiles():
    graph = build_graph()
    assert "model" in graph.get_graph().nodes


def test_single_node_answers():
    out = build_graph().invoke({"messages": [HumanMessage("Say OK")]})
    assert len(out["messages"]) == 2
    assert isinstance(out["messages"][-1], AIMessage)
