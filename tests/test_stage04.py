"""Tools are tested without the LLM and without the network: the catalogue is the fixture."""
from app.tools import search_papers, get_paper, format_citation


def test_search_finds_by_alias_and_by_title():
    assert search_papers.invoke({"query": "word2vec"})["results"][0]["arxiv_id"] == "1301.3781"
    out = search_papers.invoke({"query": "low-rank adaptation of large language models", "max_results": 3})
    assert out["results"][0]["arxiv_id"] == "2106.09685"
    assert out["count"] == 3 and all(r["source"] == "catalogue" for r in out["results"])


def test_get_paper_normalises_ids_and_rejects_titles():
    p = get_paper.invoke({"arxiv_id": "https://arxiv.org/abs/1706.03762v7"})
    assert p["title"] == "Attention Is All You Need" and len(p["authors"]) == 8 and p["year"] == 2017
    assert "error" in get_paper.invoke({"arxiv_id": "Attention Is All You Need"})
    assert "error" in get_paper.invoke({"arxiv_id": "2501.99999"})     # valid id, not in the catalogue


def test_format_citation_is_built_from_the_record():
    apa = format_citation.invoke({"arxiv_id": "1706.03762"})["citation"]
    assert apa.startswith("Vaswani, A., Shazeer, N.,") and "(2017)" in apa
    bib = format_citation.invoke({"arxiv_id": "1810.04805", "style": "bibtex"})["citation"]
    assert bib.startswith("@article{devlin2018bert,") and "Jacob Devlin and Ming-Wei Chang" in bib
    llama2 = format_citation.invoke({"arxiv_id": "2307.09288"})["citation"]
    assert "... Scialom, T." in llama2          # APA 7: 68 authors -> first 19, ellipsis, last
