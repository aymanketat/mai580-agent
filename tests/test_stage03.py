import pytest
from langchain_core.messages import HumanMessage
from app.graph import build_graph


@pytest.mark.parametrize("text,intent", [
    ("Give me the BibTeX entry for 'Attention Is All You Need'.", "citation"),
    ("What are the key papers on parameter-efficient fine-tuning of large language models?",
     "literature_search"),
])
@pytest.mark.live
def test_router(text, intent):
    out = build_graph().invoke({"messages": [HumanMessage(text)]})
    assert out["intent"] == intent
