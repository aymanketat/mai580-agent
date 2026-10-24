"""Shared test setup.

- Loads .env so live tests find the key; tests that need the model are marked `live` and are
  skipped when no real OPENAI_API_KEY is configured (a placeholder key is set so that graphs can
  still be *built* offline: constructing ChatOpenAI needs a key string but makes no request).
- Redirects the Semantic Scholar disk cache and snapshot to a temporary directory for every test,
  so tests never read or write data/cache/ or data/citations_snapshot.json.
"""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()
_PLACEHOLDER = "sk-offline-tests-no-real-key"
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("sk-..."):
    os.environ["OPENAI_API_KEY"] = _PLACEHOLDER

import app.tools as tools   # noqa: E402  (imports no model)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("OPENAI_API_KEY") == _PLACEHOLDER:
        skip = pytest.mark.skip(reason="live test: no real OPENAI_API_KEY configured")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_citation_cache(tmp_path, monkeypatch):
    """Offline tests must not depend on the user's .env: arXiv stays off, the Semantic Scholar
    flag is on (tests that exercise it stub `_s2_get` or set S2_LIVE=0 themselves), the TTL is the
    default, and the cache and snapshot live in a temporary directory."""
    monkeypatch.setenv("ARXIV_LIVE", "0")
    monkeypatch.setenv("S2_LIVE", "1")
    monkeypatch.delenv("S2_CACHE_TTL_HOURS", raising=False)
    monkeypatch.setattr(tools, "S2_CACHE_DIR", tmp_path / "s2cache")
    monkeypatch.setattr(tools, "S2_SNAPSHOT_PATH", tmp_path / "no_snapshot.json")
    monkeypatch.setattr(tools, "_SNAPSHOT", None)
    tools._S2_CACHE.clear()
    yield
    tools._S2_CACHE.clear()
