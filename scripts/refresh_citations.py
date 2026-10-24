"""Refresh data/citations_snapshot.json with real Semantic Scholar responses for the catalogue.

    python scripts/refresh_citations.py

The snapshot is a committed, dated copy of public API responses so that demos work when the API
is down or rate-limited. get_citation_count uses it only when neither a fresh nor a stale
runtime cache entry exists, and labels the value source="snapshot" with its retrieval time.
Papers that fail now (rate limit, not found) keep their previous dated value if one exists and
are otherwise absent; nothing is invented. Run again later to fill the gaps.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import tools                                     # noqa: E402

OUT = tools.S2_SNAPSHOT_PATH

if __name__ == "__main__":
    existing = {}
    try:
        loaded = json.loads(OUT.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("papers"), dict):
            existing = loaded["papers"]
    except (OSError, ValueError):
        pass
    papers, failed = dict(existing), []
    for arxiv_id in tools.CATALOGUE:
        data = tools._s2_get(arxiv_id)
        if "error" in data:
            failed.append((arxiv_id, data["error"]))
        else:
            papers[arxiv_id] = {"arxiv_id": arxiv_id, "retrieved_at": tools._iso(tools._now()), "data": data}
            print(f"{arxiv_id}: {data.get('citationCount')} citations")
        time.sleep(1.5)                                   # be polite to the shared, unauthenticated pool
    OUT.write_text(json.dumps({
        "source": "Semantic Scholar Graph API v1, https://api.semanticscholar.org/graph/v1/paper/arXiv:<id>",
        "fields": "title,year,citationCount,influentialCitationCount",
        "note": "Real public responses copied verbatim with their retrieval times; refreshed by scripts/refresh_citations.py.",
        "papers": papers,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(papers)} papers to {OUT}; failed now: {failed}")
