"""Tools (Stage 4+).

The Research Assistant's data is the paper catalogue in data/papers.json (built from the arXiv
API by scripts/build_catalogue.py). Tools read it first and only touch the network when
ARXIV_LIVE=1, so tests and demos run offline.

Every tool returns data, never raises: ToolNode's default error handler only converts bad
arguments into a ToolMessage and re-raises everything else, which would abort the whole run.
"""
import json
import os
import re
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from app import arxiv_client

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_PATH = ROOT / "data" / "papers.json"
CATALOGUE: dict[str, dict] = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
_LIVE: dict[str, dict] = {}      # records fetched from arXiv during this process

# Short names that appear in neither the title nor the abstract of their paper.
ALIASES = {
    "1706.03762": ["transformer"], "1810.04805": ["bert"], "2005.14165": ["gpt-3", "gpt3"],
    "2010.11929": ["vit", "vision transformer"], "2103.00020": ["clip"], "1512.03385": ["resnet"],
    "1412.6980": ["adam"], "1312.6114": ["vae"], "1406.2661": ["gan", "gans"],
    "1301.3781": ["word2vec"], "1910.10683": ["t5"], "1908.10084": ["sbert", "sentence-bert"],
    "2005.11401": ["rag"], "2106.09685": ["lora"], "2203.02155": ["instructgpt", "rlhf"],
    "2302.13971": ["llama"], "2307.09288": ["llama 2", "llama2"], "2201.11903": ["cot", "chain-of-thought"],
    "2210.03629": ["react"], "2305.10601": ["tot", "tree of thoughts"], "2303.11366": ["reflexion"],
    "2302.04761": ["toolformer"], "1409.0473": ["bahdanau attention"],
}
STOPWORDS = {"a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with", "is", "are",
             "paper", "papers", "model", "models", "learning"}
ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def _live() -> bool:
    return os.getenv("ARXIV_LIVE", "0") == "1"


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in STOPWORDS}


def normalize_id(raw: str) -> str | None:
    """'arXiv:1706.03762v7' or 'https://arxiv.org/abs/1706.03762' -> '1706.03762'; None if not an id."""
    m = ID_RE.search(raw or "")
    return m.group(1) if m else None


def _record(arxiv_id: str) -> dict | None:
    rec = CATALOGUE.get(arxiv_id) or _LIVE.get(arxiv_id)
    if rec is None and _live():
        try:
            for p in arxiv_client.fetch_by_ids([arxiv_id]):
                _LIVE[p["arxiv_id"]] = p
        except Exception as exc:                          # network, parse: report, do not raise
            return {"error": f"arXiv could not be reached: {exc}"}
        rec = _LIVE.get(arxiv_id)
    return rec


def _card(p: dict, source: str) -> dict:
    return {"arxiv_id": p["arxiv_id"], "title": p["title"], "first_author": p["authors"][0],
            "author_count": len(p["authors"]), "year": p["year"],
            "snippet": p["abstract"][:200], "source": source}


# ---------------------------------------------------------------- Stage 4
@tool
def search_papers(query: str, max_results: int = 5) -> dict:
    """Find machine-learning papers by keywords, method name, partial title or author surname.
    Searches the local catalogue first (lexical match on title, abstract, authors and short names
    like BERT, ViT, word2vec); use words a paper would use. Returns compact cards with the arxiv_id
    to pass to get_paper or format_citation. count 0 means the catalogue has nothing relevant."""
    q = _tokens(query)
    phrase = " ".join(query.lower().split())
    scored = []
    for arxiv_id, p in CATALOGUE.items():
        title, abstract = _tokens(p["title"]), _tokens(p["abstract"])
        authors = _tokens(" ".join(p["authors"]))
        aliases = _tokens(" ".join(ALIASES.get(arxiv_id, [])))
        score = sum(3 * (t in title) + 2 * (t in aliases) + 2 * (t in authors) + (t in abstract) for t in q)
        if phrase and phrase in p["title"].lower():
            score += 4
        if score:
            scored.append((score, p["year"], p))
    scored.sort(key=lambda s: (-s[0], -s[1]))
    n = max(1, min(int(max_results), 10))
    results = [_card(p, "catalogue") for _, _, p in scored[:n]]
    if not results and _live():
        try:
            for p in arxiv_client.search(query, n):
                _LIVE[p["arxiv_id"]] = p
                results.append(_card(p, "arxiv"))
        except Exception as exc:
            return {"query": query, "count": 0, "results": [], "error": f"arXiv search failed: {exc}"}
    return {"query": query, "count": len(results), "results": results}


@tool
def get_paper(arxiv_id: str) -> dict:
    """Return the full record of ONE paper by arXiv id (e.g. "1706.03762"; "arXiv:" prefixes,
    version suffixes and arxiv.org URLs are accepted): title, complete author list, published date,
    year, categories, abstract, url. Use this, not memory, for author lists and abstracts.
    Does not accept titles: call search_papers first to get the id."""
    bare = normalize_id(arxiv_id)
    if bare is None:
        return {"error": f"'{arxiv_id}' is not an arXiv id", "hint": "use search_papers to find the id first"}
    rec = _record(bare)
    if rec is None:
        return {"error": f"No paper {bare} in the catalogue" + ("" if _live() else " (offline)")}
    return rec


def _surname_initials(name: str) -> tuple[str, str]:
    parts = name.replace(".", " ").split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[-1], " ".join(f"{p[0]}." for p in parts[:-1])


def _apa_authors(names: list[str]) -> str:
    fmt = [f"{s}, {i}".rstrip(", ") for s, i in map(_surname_initials, names)]
    if len(fmt) > 20:                                     # APA 7: first 19, ellipsis, last
        return ", ".join(fmt[:19]) + ", ... " + fmt[-1]
    if len(fmt) == 1:
        return fmt[0]
    return ", ".join(fmt[:-1]) + ", & " + fmt[-1]


@tool
def format_citation(arxiv_id: str, style: Literal["apa", "bibtex", "ieee"] = "apa") -> dict:
    """Build a reference for ONE paper identified by arXiv id, in APA 7, BibTeX or IEEE style.
    The text is generated from the paper record, never from memory: copy it verbatim and do not
    edit author names. If you only know a title, call search_papers first to get the id."""
    bare = normalize_id(arxiv_id)
    if bare is None:
        return {"error": f"'{arxiv_id}' is not an arXiv id", "hint": "use search_papers to find the id first"}
    p = _record(bare)
    if p is None or "error" in p:
        return p or {"error": f"No paper {bare} in the catalogue"}
    first_surname, _ = _surname_initials(p["authors"][0])
    if style == "bibtex":
        words = [t for t in re.split(r"[^a-z0-9]+", p["title"].lower()) if t and t not in STOPWORDS]
        key = re.sub(r"[^a-z0-9]", "", f"{first_surname.lower()}{p['year']}{words[0] if words else 'paper'}")
        citation = (f"@article{{{key},\n  title={{{p['title']}}},\n"
                    f"  author={{{' and '.join(p['authors'])}}},\n"
                    f"  journal={{arXiv preprint arXiv:{bare}}},\n  year={{{p['year']}}},\n"
                    f"  url={{{p['url']}}}\n}}")
    elif style == "ieee":
        names = [f"{i} {s}".strip() for s, i in map(_surname_initials, p["authors"])]
        authors = ", ".join(names[:6]) + (" et al." if len(names) > 6 else "")
        citation = f'{authors}, "{p["title"]}," arXiv:{bare}, {p["year"]}.'
    else:
        citation = f"{_apa_authors(p['authors'])} ({p['year']}). {p['title']}. arXiv:{bare}. {p['url']}"
    return {"arxiv_id": bare, "style": style, "citation": citation}


TOOLS = [search_papers, get_paper, format_citation]
