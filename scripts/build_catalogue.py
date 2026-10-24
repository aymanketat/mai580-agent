"""Build data/papers.json from the arXiv API (no key needed).

The catalogue is the Research Assistant's local, cached corpus: the tools in
app/tools.py read it first and only call arXiv for papers that are not in it.
Run again to refresh:  python scripts/build_catalogue.py
"""
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
OUT = Path(__file__).resolve().parents[1] / "data" / "papers.json"

# Well-known machine-learning papers; the metadata comes from arXiv, not from memory.
IDS = [
    "1706.03762", "1810.04805", "2005.14165", "1512.03385", "1412.6980",
    "2010.11929", "2103.00020", "1406.2661", "1312.6114", "2210.03629",
    "2201.11903", "2005.11401", "2302.13971", "1908.10084", "2106.09685",
    "2203.02155", "1301.3781", "1409.0473", "2307.09288", "1910.10683",
    "2305.10601", "2303.11366", "2302.04761",
]


def parse_entry(entry: ET.Element) -> dict:
    """Turn one Atom <entry> into a plain dict."""
    text = lambda tag: (entry.findtext(f"a:{tag}", default="", namespaces=ATOM) or "").strip()
    raw_id = text("id")                                   # http://arxiv.org/abs/1706.03762v7
    arxiv_id = raw_id.rsplit("/", 1)[-1]
    version = arxiv_id.rsplit("v", 1)[-1] if "v" in arxiv_id else ""
    return {
        "arxiv_id": arxiv_id.split("v")[0],
        "version": f"v{version}" if version else "",
        "title": " ".join(text("title").split()),
        "authors": [a.findtext("a:name", namespaces=ATOM).strip()
                    for a in entry.findall("a:author", ATOM)],
        "published": text("published")[:10],
        "year": int(text("published")[:4]),
        "primary_category": entry.find("arxiv:primary_category", ATOM).get("term"),
        "categories": [c.get("term") for c in entry.findall("a:category", ATOM)],
        "abstract": " ".join(text("summary").split()),
        "url": f"https://arxiv.org/abs/{arxiv_id.split('v')[0]}",
    }


def fetch(ids: list[str]) -> list[dict]:
    url = f"{ARXIV_API}?id_list={','.join(ids)}&max_results={len(ids)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        root = ET.fromstring(r.read())
    return [parse_entry(e) for e in root.findall("a:entry", ATOM)]


if __name__ == "__main__":
    papers = {}
    for i in range(0, len(IDS), 10):                      # arXiv asks for small, spaced requests
        for p in fetch(IDS[i:i + 10]):
            papers[p["arxiv_id"]] = p
        time.sleep(3)
    OUT.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(papers)} papers to {OUT}")
