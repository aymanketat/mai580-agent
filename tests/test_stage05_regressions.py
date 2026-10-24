"""Regression tests for the Stage 5 tool fixes (review of commit 04b2d4f). All offline: no model,
no network. Covers reference verification, author-name forms, Retry-After parsing, the persistent
citation-count cache, and the step budget in scripts/inspect_graph.py."""
import email.message
import importlib.util
import json
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

import app.tools as tools
from app.state import State
from app.tools import check_citation, get_citation_count

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 1. reference verification
def test_invented_subtitle_is_not_verified():
    out = check_citation.invoke({"title": "ReAct: A Completely Fabricated Study of Ocean Currents",
                                 "first_author": "Yao", "year": 2022})
    assert out["verdict"] == "mismatch"
    assert out["candidate"]["arxiv_id"] == "2210.03629"          # the likely paper is still reported
    assert "title" in [m["field"] for m in out["mismatches"]]
    assert out["title_match"] == "different"


def test_near_title_is_not_verified():
    out = check_citation.invoke({"title": "Attention Is All You Hate", "first_author": "Vaswani", "year": 2017})
    assert out["verdict"] == "mismatch" and out["title_match"] == "different"
    assert out["candidate"]["title"] == "Attention Is All You Need"
    assert [m["field"] for m in out["mismatches"]] == ["title"]


@pytest.mark.parametrize("title", [
    "attention is all you need",                 # case
    "Attention is all you need!",                # punctuation
    "An Image is Worth 16x16 Words",             # subtitle omitted
    "ReAct:",                                    # head title with a trailing colon
    "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale",   # × vs x
])
def test_legitimate_title_variants_are_verified(title):
    out = check_citation.invoke({"title": title})
    assert out["verdict"] == "verified", out
    assert out["title_match"] in {"exact", "subtitle_omitted"}


# ---------------------------------------------------------------- 2. author-name normalisation
@pytest.mark.parametrize("author", ["Vaswani", "Ashish Vaswani", "Vaswani, A.", "A. Vaswani",
                                    "Vaswani et al.", "VASWANI", "Vaswani A", "A.Vaswani",
                                    "Vaswani et al. (2017)", "Vaswani et. al.", "Vaswani Jr., A.",
                                    "Ashish Vaswani and Noam Shazeer", "Vaswani AS", "Vaswani Ashish"])
def test_first_author_forms_match(author):
    out = check_citation.invoke({"title": "Attention Is All You Need", "first_author": author, "year": 2017})
    assert out["verdict"] == "verified", out["mismatches"]


@pytest.mark.parametrize("title", ["Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
                                   "Sentence BERT: Sentence Embeddings using Siamese BERT Networks",
                                   "SentenceBERT"])
def test_hyphenation_variants_are_verified(title):
    assert check_citation.invoke({"title": title, "first_author": "Reimers"})["verdict"] == "verified"


@pytest.mark.parametrize("author", ["Bengio", "Shazeer", "Noam Shazeer", "Devlin, J.", "Noam Shazeer and Ashish Vaswani"])
def test_different_first_author_is_a_mismatch(author):
    out = check_citation.invoke({"title": "Attention Is All You Need", "first_author": author})
    assert out["verdict"] == "mismatch"
    assert [m["field"] for m in out["mismatches"]] == ["first_author"]


def test_author_folding_handles_accents_and_hyphens():
    assert tools._author_surname("Touvrón, H.") == tools._author_surname("Hugo Touvron") == "touvron"
    assert tools._author_surname("Pouget-Abadie, Jean") == tools._author_surname("Jean Pouget-Abadie")


# ---------------------------------------------------------------- 5. Retry-After
def test_retry_after_values():
    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)
    assert tools._parse_retry_after("3", now) == 3.0
    assert tools._parse_retry_after("120", now) == tools.S2_RETRY_CAP_S          # bounded
    future = email.utils.format_datetime(now + timedelta(seconds=4))
    assert tools._parse_retry_after(future, now) == pytest.approx(4.0, abs=1.0)
    assert tools._parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now) == 0.0  # past date
    assert tools._parse_retry_after("garbage", now) == tools.S2_RETRY_DEFAULT_S
    assert tools._parse_retry_after(None, now) == tools.S2_RETRY_DEFAULT_S
    assert tools._parse_retry_after("", now) == tools.S2_RETRY_DEFAULT_S


def _http_error(code: str | int, retry_after: str | None = None):
    hdr = email.message.Message()
    if retry_after is not None:
        hdr["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://api.semanticscholar.org/x", int(code), "err", hdr, None)


def test_rate_limit_retries_once_with_http_date_then_structured_error(monkeypatch):
    attempts, sleeps = [], []
    def urlopen(req, timeout=10):
        attempts.append(req.full_url)
        raise _http_error(429, "Wed, 21 Oct 2015 07:28:00 GMT")         # past HTTP-date: no ValueError
    monkeypatch.setattr(tools.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)
    out = tools._s2_get("1706.03762")
    assert out["error"] == "rate_limited" and "detail" in out
    assert len(attempts) == 1 + tools.S2_MAX_RETRIES and sleeps == [0.0]


def test_rate_limit_then_success(monkeypatch):
    calls, sleeps = [], []
    def urlopen(req, timeout=10):
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(429, "2")
        class R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps({"title": "T", "year": 2017, "citationCount": 5}).encode()
        return R()
    monkeypatch.setattr(tools.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)
    assert tools._s2_get("1706.03762")["citationCount"] == 5 and sleeps == [2.0]


def test_not_found_and_connection_errors_are_structured(monkeypatch):
    monkeypatch.setattr(tools.urllib.request, "urlopen", lambda req, timeout=10: (_ for _ in ()).throw(_http_error(404)))
    assert tools._s2_get("1706.03762")["error"] == "not_found"
    monkeypatch.setattr(tools.urllib.request, "urlopen", lambda req, timeout=10: (_ for _ in ()).throw(OSError("dns")))
    assert tools._s2_get("1706.03762")["error"] == "unavailable"


# ---------------------------------------------------------------- 4. persistent cache
STUB = {"title": "Attention Is All You Need", "year": 2017, "citationCount": 123456, "influentialCitationCount": 7}


def test_cache_persists_across_restart_and_keeps_original_time(monkeypatch):
    monkeypatch.setenv("S2_LIVE", "1")
    monkeypatch.setattr(tools, "_s2_get", lambda aid: dict(STUB))
    first = get_citation_count.invoke({"arxiv_id": "1706.03762"})
    assert first["source"] == "semantic_scholar" and first["stale"] is False
    assert (tools.S2_CACHE_DIR / "1706.03762.json").exists()
    tools._S2_CACHE.clear()                                            # simulate a restart
    monkeypatch.setattr(tools, "_s2_get", lambda aid: pytest.fail("network must not be called"))
    again = get_citation_count.invoke({"arxiv_id": "1706.03762"})
    assert again["source"] == "cache" and again["stale"] is False
    assert again["retrieved_at"] == first["retrieved_at"]              # original time preserved
    assert again["citation_count"] == 123456


def test_expired_cache_is_marked_stale_and_offline_mode_uses_it(monkeypatch):
    old = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tools._cache_write("1706.03762", {"arxiv_id": "1706.03762", "retrieved_at": old, "data": dict(STUB)})
    tools._S2_CACHE.clear()
    monkeypatch.setenv("S2_LIVE", "0")
    out = get_citation_count.invoke({"arxiv_id": "1706.03762"})
    assert out["source"] == "cache" and out["stale"] is True and out["retrieved_at"] == old
    assert "offline" in out["note"]
    monkeypatch.setenv("S2_LIVE", "1")
    monkeypatch.setattr(tools, "_s2_get", lambda aid: {"error": "rate_limited", "detail": "x"})
    out = get_citation_count.invoke({"arxiv_id": "1706.03762"})
    assert out["source"] == "cache" and out["stale"] is True and "rate_limited" in out["note"]


def test_errors_are_not_cached_and_offline_without_cache_is_an_error(monkeypatch):
    monkeypatch.setenv("S2_LIVE", "1")
    monkeypatch.setattr(tools, "_s2_get", lambda aid: {"error": "unavailable", "detail": "x"})
    out = get_citation_count.invoke({"arxiv_id": "1810.04805"})
    assert out["error"] == "unavailable" and not (tools.S2_CACHE_DIR / "1810.04805.json").exists()
    monkeypatch.setenv("S2_LIVE", "0")
    assert get_citation_count.invoke({"arxiv_id": "1810.04805"})["error"] == "offline"


@pytest.mark.parametrize("content", ["{not json", '{"retrieved_at": 123, "data": {"citationCount": 9}}',
                                     '{"retrieved_at": null, "data": {}}', "[]", '"x"'])
def test_damaged_cache_file_is_ignored(monkeypatch, content):
    tools.S2_CACHE_DIR.mkdir(parents=True)
    (tools.S2_CACHE_DIR / "2106.09685.json").write_text(content, encoding="utf-8")
    monkeypatch.setenv("S2_LIVE", "1")
    monkeypatch.setattr(tools, "_s2_get", lambda aid: dict(STUB))
    out = get_citation_count.invoke({"arxiv_id": "2106.09685"})
    assert out["source"] == "semantic_scholar"
    assert json.loads((tools.S2_CACHE_DIR / "2106.09685.json").read_text())["data"]["citationCount"] == 123456


@pytest.mark.parametrize("content", ["[]", '"abc"', "null", '{"papers": "x"}', '{"papers": {"1301.3781": {"retrieved_at": 5, "data": {}}}}'])
def test_damaged_snapshot_is_ignored(monkeypatch, content):
    tools.S2_SNAPSHOT_PATH.write_text(content, encoding="utf-8")
    monkeypatch.setenv("S2_LIVE", "0")
    assert get_citation_count.invoke({"arxiv_id": "1301.3781"})["error"] == "offline"


def test_bad_ttl_setting_and_bad_response_bodies_do_not_raise(monkeypatch):
    monkeypatch.setenv("S2_CACHE_TTL_HOURS", "")
    assert tools._s2_ttl_s() == 168 * 3600
    monkeypatch.setenv("S2_CACHE_TTL_HOURS", "abc")
    assert tools._s2_ttl_s() == 168 * 3600
    monkeypatch.setenv("S2_CACHE_TTL_HOURS", "-5")
    assert tools._s2_ttl_s() == 0
    for body in [b"null", b"[]", b'"Too Many Requests"', b'{"message": "x"}', b'{"title": "T"}']:
        class R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, _b=body): return _b
        monkeypatch.setattr(tools.urllib.request, "urlopen", lambda req, timeout=10: R())
        assert tools._s2_get("1706.03762")["error"] == "unavailable"
    monkeypatch.setenv("S2_LIVE", "1")
    monkeypatch.setattr(tools, "_s2_get", lambda aid: {"title": "T"})     # success-shaped, no count
    out = get_citation_count.invoke({"arxiv_id": "1706.03762"})
    assert out["error"] == "unavailable" and not (tools.S2_CACHE_DIR / "1706.03762.json").exists()


def test_snapshot_is_used_last_and_labelled(monkeypatch):
    snap = {"papers": {"1301.3781": {"arxiv_id": "1301.3781", "retrieved_at": "2026-10-24T00:00:00Z",
                                     "data": {"title": "w2v", "year": 2013, "citationCount": 42}}}}
    tools.S2_SNAPSHOT_PATH.write_text(json.dumps(snap), encoding="utf-8")
    monkeypatch.setenv("S2_LIVE", "0")
    out = get_citation_count.invoke({"arxiv_id": "1301.3781"})
    assert out["source"] == "snapshot" and out["citation_count"] == 42 and out["retrieved_at"] == "2026-10-24T00:00:00Z"


# ---------------------------------------------------------------- 3. step budget in inspect_graph
def _load_inspect_module():
    spec = importlib.util.spec_from_file_location("inspect_graph", ROOT / "scripts" / "inspect_graph.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inspection_stops_at_the_step_budget():
    """A stubbed agent that always calls a tool never terminates; the budget must stop it."""
    @tool
    def ping(x: str) -> str:
        """Return pong."""
        return "pong"

    def always_call_a_tool(state: State) -> dict:
        n = len(state["messages"])
        return {"messages": [AIMessage(content="", tool_calls=[{"name": "ping", "args": {"x": "a"}, "id": f"call-{n}"}])]}

    builder = StateGraph(State)
    builder.add_node("agent", always_call_a_tool)
    builder.add_node("tools", ToolNode([ping]))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    endless = builder.compile()

    inspect = _load_inspect_module()
    lines = []
    result = inspect.stream_run(endless, "loop forever", recursion_limit=6, out=lines.append)
    assert result["budget_exceeded"] is True
    assert result["answer"] == inspect.STEP_BUDGET_MESSAGE
    assert result["steps"] == 6                                        # one node update per step
    assert any("step budget of 6 reached" in line for line in lines)
    assert any("--- tools ---" in line for line in lines) and any("pong" in line for line in lines)  # the loop really ran
    assert inspect.stream_run.__defaults__[0] == inspect.RECURSION_LIMIT  # default budget = run.py's
