from langchain_core.messages import AIMessage, HumanMessage
from app.graph import build_graph


def test_single_node_answers():
    # One question in, one answer out. Later stages add nodes but must keep this contract.
    out = build_graph().invoke({"messages": [HumanMessage("Say OK")]})
    assert len(out["messages"]) == 2
    assert isinstance(out["messages"][-1], AIMessage)
