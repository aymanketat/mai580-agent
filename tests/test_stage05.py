import pytest
from langchain_core.messages import HumanMessage

import app.tools as tools
from app.graph import RECURSION_LIMIT, build_graph
from app.tools import TOOLS, check_citation, get_citation_count


def test_loop_uses_two_tools():
    out = build_graph().invoke(
        {"messages": [HumanMessage("Find the paper that introduced Sentence-BERT and give me its BibTeX entry.")]},
        {"recursion_limit": RECURSION_LIMIT})
    called = [c["name"] for m in out["messages"] for c in getattr(m, "tool_calls", [])]
    assert "search_papers" in called and "format_citation" in called


def test_graph_closes_the_loop():
    g = build_graph().get_graph()
    assert ("tools", "agent") in {(e.source, e.target) for e in g.edges}   # the edge that makes it an agent
    assert "final_answer" not in g.nodes


# ---- offline tool tests (no LLM, no network) ----
@pytest.mark.parametrize("title,author,year,verdict,field", [
    ("Attention Is All You Need", "Vaswani", 2017, "verified", None),
    ("attention is all you need", None, None, "verified", None),
    ("ReAct: Synergizing Reasoning and Acting in Language Models", "Yao", 2023, "mismatch", "year"),
    ("Generative Adversarial Networks", "Bengio", 2014, "mismatch", "first_author"),
    ("Scaling Laws for Neural Language Models", "Kaplan", 2020, "unverified", None),
])
def test_check_citation(title, author, year, verdict, field):
    out = check_citation.invoke({"title": title, "first_author": author, "year": year})
    assert out["verdict"] == verdict
    if field:
        assert field in [m["field"] for m in out["mismatches"]]
    if verdict != "unverified":
        assert out["arxiv_id"] in tools.CATALOGUE


def test_get_citation_count_is_stubbed_and_cached(monkeypatch):
    tools._S2_CACHE.clear()
    calls = []
    monkeypatch.setattr(tools, "_s2_get", lambda aid: calls.append(aid) or {
        "title": "Attention Is All You Need", "year": 2017, "citationCount": 123456, "influentialCitationCount": 7})
    out = get_citation_count.invoke({"arxiv_id": "arXiv:1706.03762v7"})
    assert out["citation_count"] == 123456 and out["source"] == "semantic_scholar"
    assert get_citation_count.invoke({"arxiv_id": "1706.03762"})["source"] == "cache"
    assert calls == ["1706.03762"]                                  # second call served from the cache
    monkeypatch.setattr(tools, "_s2_get", lambda aid: {"error": "rate_limited"})
    assert get_citation_count.invoke({"arxiv_id": "1810.04805"})["error"] == "rate_limited"
    assert [t.name for t in TOOLS] == ["search_papers", "get_paper", "format_citation",
                                       "get_citation_count", "check_citation"]
