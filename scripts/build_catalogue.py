"""Build data/papers.json from the arXiv API (no key needed).

The catalogue is the Research Assistant's local, cached corpus: the tools in app/tools.py read
it and only call arXiv when ARXIV_LIVE=1. Run again to refresh:  python scripts/build_catalogue.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import arxiv_client                              # noqa: E402  (shared parser + throttle)

OUT = Path(__file__).resolve().parents[1] / "data" / "papers.json"

# Well-known machine-learning papers; the metadata comes from arXiv, not from memory.
IDS = [
    "1706.03762", "1810.04805", "2005.14165", "1512.03385", "1412.6980",
    "2010.11929", "2103.00020", "1406.2661", "1312.6114", "2210.03629",
    "2201.11903", "2005.11401", "2302.13971", "1908.10084", "2106.09685",
    "2203.02155", "1301.3781", "1409.0473", "2307.09288", "1910.10683",
    "2305.10601", "2303.11366", "2302.04761",
]

if __name__ == "__main__":
    papers = {}
    for i in range(0, len(IDS), 10):                      # small requests; the client spaces them 3 s apart
        for p in arxiv_client.fetch_by_ids(IDS[i:i + 10]):
            papers[p["arxiv_id"]] = p
    OUT.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(papers)} papers to {OUT}")
