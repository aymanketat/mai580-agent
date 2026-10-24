"""Thin client for the arXiv Atom API (no key needed).

Shared by scripts/build_catalogue.py (builds data/papers.json) and app/tools.py (live
fallback). arXiv asks for one request every 3 seconds, so every call goes through one
throttle; ToolNode may run tool calls in parallel threads, hence the lock.
"""
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
MIN_INTERVAL_S = 3.0

_lock = threading.Lock()
_last_call = 0.0


def _get(url: str, timeout: float = 20) -> bytes:
    global _last_call
    with _lock:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": "mai580-agent/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(4_000_000)
        _last_call = time.monotonic()
    return body


def parse_entry(entry: ET.Element) -> dict:
    """Turn one Atom <entry> into a plain dict (the record shape used in data/papers.json)."""
    text = lambda tag: (entry.findtext(f"a:{tag}", default="", namespaces=ATOM) or "").strip()
    raw_id = text("id")                                   # http://arxiv.org/abs/1706.03762v7
    versioned = raw_id.rsplit("/", 1)[-1]
    bare, _, version = versioned.partition("v")
    primary = entry.find("arxiv:primary_category", ATOM)
    return {
        "arxiv_id": bare,
        "version": f"v{version}" if version else "",
        "title": " ".join(text("title").split()),
        "authors": [a.findtext("a:name", namespaces=ATOM).strip()
                    for a in entry.findall("a:author", ATOM)],
        "published": text("published")[:10],
        "year": int(text("published")[:4]) if text("published") else None,
        "primary_category": primary.get("term") if primary is not None else None,
        "categories": [c.get("term") for c in entry.findall("a:category", ATOM)],
        "abstract": " ".join(text("summary").split()),
        "url": f"https://arxiv.org/abs/{bare}",
    }


def _query(params: dict) -> list[dict]:
    root = ET.fromstring(_get(f"{API}?{urllib.parse.urlencode(params)}"))
    entries = root.findall("a:entry", ATOM)
    # arXiv reports a bad id_list as a single entry titled "Error".
    return [parse_entry(e) for e in entries
            if (e.findtext("a:title", default="", namespaces=ATOM) or "").strip() != "Error"]


def fetch_by_ids(ids: list[str]) -> list[dict]:
    return _query({"id_list": ",".join(ids), "max_results": len(ids)})


def search(query: str, max_results: int = 5) -> list[dict]:
    return _query({"search_query": f"all:{query}", "start": 0,
                   "max_results": max_results, "sortBy": "relevance"})
