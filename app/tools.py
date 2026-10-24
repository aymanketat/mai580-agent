"""Tools (Stage 4+).

The Research Assistant's data is the paper catalogue in data/papers.json (built from the arXiv
API by scripts/build_catalogue.py). Tools read it first and only touch the network when a live
flag is set (ARXIV_LIVE=1 for arXiv, S2_LIVE=1 for Semantic Scholar; see README), so tests and
demos run offline.

Every tool returns data rather than raising, so the model receives structured error dicts it can
act on (retry, say what could not be retrieved). The graph additionally builds its ToolNode with
handle_tool_errors=True as a safety net for anything that still escapes.
"""
import difflib
import email.utils
import json
import os
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
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


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


def _live() -> bool:
    return _flag("ARXIV_LIVE", "0")


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


# ---------------------------------------------------------------- Stage 5: citation counts
S2_API = ("https://api.semanticscholar.org/graph/v1/paper/arXiv:{id}"
          "?fields=title,year,citationCount,influentialCitationCount")
S2_CACHE_DIR = ROOT / "data" / "cache" / "semantic_scholar"      # runtime cache, gitignored
S2_SNAPSHOT_PATH = ROOT / "data" / "citations_snapshot.json"     # committed demo snapshot (optional)
S2_RETRY_DEFAULT_S = 2.0      # when Retry-After is missing or unreadable
S2_RETRY_CAP_S = 5.0          # never sleep longer than this inside a tool call
S2_MAX_RETRIES = 1            # one retry on 429, then a structured error
_S2_CACHE: dict[str, dict] = {}                              # in-process copy of cache records
_S2_LOCK = threading.Lock()                                  # ToolNode may run tool calls in threads


def _s2_live() -> bool:
    return _flag("S2_LIVE", "1")


def _s2_ttl_s() -> int:
    """Cache lifetime in seconds from S2_CACHE_TTL_HOURS (default one week); unreadable values fall back."""
    try:
        hours = float(os.getenv("S2_CACHE_TTL_HOURS", "168") or 168)
    except ValueError:
        hours = 168.0
    return int(max(hours, 0) * 3600)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_retry_after(value: str | None, now: datetime | None = None) -> float:
    """Seconds to wait from a Retry-After header: an integer, an HTTP-date, or nothing.
    Missing or unreadable values fall back to the default; past dates mean 0; always capped."""
    if value is None or not str(value).strip():
        return S2_RETRY_DEFAULT_S
    text = str(value).strip()
    delay = None
    if re.fullmatch(r"\d+(\.\d+)?", text):
        delay = float(text)
    else:
        try:
            dt = email.utils.parsedate_to_datetime(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delay = (dt - (now or _now())).total_seconds()
        except (TypeError, ValueError, IndexError):
            delay = None
    if delay is None:
        delay = S2_RETRY_DEFAULT_S
    return max(0.0, min(delay, S2_RETRY_CAP_S))


def _s2_get(arxiv_id: str) -> dict:
    """One Semantic Scholar lookup. Returns the JSON record, or {"error": ..., "detail": ...}.
    Never raises. A 429 is retried at most S2_MAX_RETRIES times after a bounded Retry-After wait."""
    req = urllib.request.Request(S2_API.format(id=arxiv_id), headers={"User-Agent": "mai580-agent/0.1"})
    for attempt in range(S2_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                body = json.loads(r.read())
            if not isinstance(body, dict) or not isinstance(body.get("citationCount"), int):
                return {"error": "unavailable", "detail": "response without an integer citationCount"}
            return body
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"error": "not_found", "detail": f"Semantic Scholar has no record for arXiv:{arxiv_id}"}
            if exc.code == 429:
                if attempt < S2_MAX_RETRIES:
                    time.sleep(_parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None))
                    continue
                return {"error": "rate_limited",
                        "detail": f"Semantic Scholar rate limit; {S2_MAX_RETRIES} retry made; try again in a minute"}
            return {"error": "unavailable", "detail": f"HTTP {exc.code}"}
        except Exception as exc:                          # timeouts, DNS, bad JSON
            return {"error": "unavailable", "detail": f"{type(exc).__name__}: {exc}"}
    return {"error": "unavailable", "detail": "no attempt succeeded"}


def _cache_path(arxiv_id: str) -> Path:
    return S2_CACHE_DIR / f"{arxiv_id}.json"


def _cache_read(arxiv_id: str) -> dict | None:
    """A cached record {"arxiv_id", "retrieved_at", "data"}; a missing or damaged file reads as None."""
    rec = _S2_CACHE.get(arxiv_id)
    if rec is not None:
        return rec if _well_formed(rec) else None
    path = _cache_path(arxiv_id)
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):     # missing, unreadable or not JSON
        return None
    if not _well_formed(rec):
        return None
    _S2_CACHE[arxiv_id] = rec
    return rec


def _well_formed(rec) -> bool:
    """A usable record: {"retrieved_at": "<ISO string>", "data": {..., "citationCount": int}}."""
    return (isinstance(rec, dict) and isinstance(rec.get("retrieved_at"), str)
            and isinstance(rec.get("data"), dict) and isinstance(rec["data"].get("citationCount"), int))


def _cache_write(arxiv_id: str, rec: dict) -> None:
    """Atomic write (temp file + os.replace) under a lock, so concurrent tool calls cannot corrupt it.
    If the replace fails (for example a reader holds the file open on Windows) the temp file is
    removed and the in-process copy still serves this run."""
    with _S2_LOCK:
        _S2_CACHE[arxiv_id] = rec
        tmp = _cache_path(arxiv_id).with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            S2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            for attempt in range(3):
                try:
                    os.replace(tmp, _cache_path(arxiv_id))
                    break
                except PermissionError:
                    if attempt == 2:
                        raise
                    time.sleep(0.05)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


_SNAPSHOT: dict | None = None


def _snapshot_read(arxiv_id: str) -> dict | None:
    """A committed demo snapshot of real responses (scripts/refresh_citations.py); None if absent."""
    global _SNAPSHOT
    if _SNAPSHOT is None:
        try:
            loaded = json.loads(S2_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = {}
        _SNAPSHOT = loaded if isinstance(loaded, dict) else {}
    papers = _SNAPSHOT.get("papers")
    rec = papers.get(arxiv_id) if isinstance(papers, dict) else None
    return rec if _well_formed(rec) else None


def _age_s(rec: dict) -> float:
    try:
        retrieved = datetime.strptime(rec["retrieved_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return float("inf")
    return (_now() - retrieved).total_seconds()


def _count_result(arxiv_id: str, rec: dict, source: str, stale: bool, note: str | None = None) -> dict:
    d = rec["data"]
    out = {"arxiv_id": arxiv_id, "title": d.get("title"), "year": d.get("year"),
           "citation_count": d.get("citationCount"),
           "influential_citation_count": d.get("influentialCitationCount"),
           "retrieved_at": rec["retrieved_at"],          # the ORIGINAL retrieval time, never "now"
           "source": source, "stale": stale}
    if note:
        out["note"] = note
    return out


@tool
def get_citation_count(arxiv_id: str) -> dict:
    """Return how many times a paper has been cited according to Semantic Scholar. Requires the
    arXiv id: resolve titles with search_papers or check_citation first. The result carries
    retrieved_at (when the number was fetched), source ("semantic_scholar", "cache" or
    "snapshot") and stale (true when older than the cache limit): always quote the count with its
    date, and say when it is a cached or stale value. On failure returns an error (not_found,
    rate_limited, offline, unavailable), never a guessed number."""
    bare = normalize_id(arxiv_id)
    if bare is None:
        return {"error": f"'{arxiv_id}' is not an arXiv id", "hint": "use search_papers to find the id first"}
    ttl = _s2_ttl_s()
    cached = _cache_read(bare)
    if cached is not None and _age_s(cached) < ttl:
        return _count_result(bare, cached, "cache", stale=False)
    if _s2_live():
        data = _s2_get(bare)
        if not isinstance(data, dict) or "error" in data or not isinstance(data.get("citationCount"), int):
            data = (data if isinstance(data, dict) and "error" in data
                    else {"error": "unavailable", "detail": "response without an integer citationCount"})
        else:                                             # only well-formed successes are cached
            rec = {"arxiv_id": bare, "retrieved_at": _iso(_now()), "data": data}
            _cache_write(bare, rec)
            return _count_result(bare, rec, "semantic_scholar", stale=False)
        failure = f"live lookup failed ({data['error']})"
    else:
        failure = "offline mode (S2_LIVE is not 1)"
    if cached is not None:                                # expired, but better than nothing: say so
        return _count_result(bare, cached, "cache", stale=True,
                             note=f"{failure}; this is the value retrieved on {cached['retrieved_at']}")
    snap = _snapshot_read(bare)
    if snap is not None:
        return _count_result(bare, snap, "snapshot", stale=_age_s(snap) >= ttl,
                             note=f"{failure}; this is the committed snapshot value from {snap['retrieved_at']}")
    if _s2_live():
        return {"arxiv_id": bare, **data}
    return {"arxiv_id": bare, "error": "offline", "detail": "S2_LIVE is not 1 and no cached value exists"}


# ---------------------------------------------------------------- Stage 5: reference checking
def _fold(text: str) -> str:
    """Lower-case ASCII letters and digits only: 'Rozière' -> 'roziere', 'Pouget-Abadie' -> 'pougetabadie'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _norm_title(text: str) -> str:
    text = text.replace("×", "x")                         # before the ASCII fold, which would drop it
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(t for t in re.split(r"[^a-z0-9]+", text.lower()) if t)


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _author_tokens(name: str) -> tuple[list[str], bool]:
    """Name tokens that could be a surname (initials, suffixes, years, 'et al.' removed), and
    whether the name was written surname-first with a comma ('Vaswani, A.')."""
    s = re.sub(r"\(.*?\)", " ", name)                      # drop parenthesised years etc.
    s = re.sub(r"\bet\.?\s*al\.?", " ", s, flags=re.I)
    s = re.split(r"\s+and\s+|&|;", s, maxsplit=1)[0]      # first author of a BibTeX-style list
    surname_first = "," in s
    if surname_first:                                     # "Vaswani, A."
        s = s.split(",", 1)[0]
    tokens = [t for t in re.split(r"\s+", s.replace(".", ". ").strip()) if t]
    tokens = [t for t in tokens
              if re.search(r"[A-Za-z]", t)                                  # has letters
              and not re.fullmatch(r"[A-Za-z]\.?", t)                       # not an initial
              and not re.fullmatch(r"[A-Z]{2,3}", t)                        # not Vancouver initials "TB"
              and _fold(t) not in _SUFFIXES]
    return tokens, surname_first


def _author_surname(name: str) -> str:
    """Folded surname from 'Vaswani', 'Ashish Vaswani', 'Vaswani, A.', 'A. Vaswani', 'A.Vaswani',
    'Vaswani et al.', 'Vaswani et al. (2017)' or 'Brown TB'; suffixes such as 'Jr.' are ignored."""
    tokens, _ = _author_tokens(name)
    return _fold(tokens[-1]) if tokens else _fold(name)


def _author_matches(given: str, canonical: str) -> bool:
    """Does the given first author refer to the catalogue's first author? Surname comparison,
    plus: a name written surname-first without a comma ('He Kaiming') matches when any token
    equals the catalogue surname."""
    target = _author_surname(canonical)
    if not target:
        return True
    tokens, surname_first = _author_tokens(given)
    if not tokens:
        return True                                       # nothing usable to compare
    if _fold(tokens[-1]) == target:
        return True
    return not surname_first and any(_fold(t) == target for t in tokens)


def _title_match(given: str, canonical: str) -> tuple[str, float]:
    """How the supplied title relates to the catalogue title: 'exact' (same after normalising case,
    punctuation, accents and 16x16/16×16), 'subtitle_omitted' (the part before the colon, given
    without any subtitle), or 'different' (anything else, however similar)."""
    given = given.strip().rstrip(":").strip()            # "ReAct:" is the head title, not a subtitle
    ng, nc = _norm_title(given), _norm_title(canonical)
    similarity = difflib.SequenceMatcher(None, ng, nc).ratio()
    squash = lambda s: s.replace(" ", "")                 # "Sentence-BERT" == "Sentence BERT" == "SentenceBERT"
    if squash(ng) == squash(nc):
        return "exact", 1.0
    if ":" not in given and ":" in canonical and squash(ng) == squash(_norm_title(canonical.split(":")[0])):
        return "subtitle_omitted", 1.0
    return "different", similarity


def _candidate_score(given: str, canonical: str) -> float:
    """For FINDING a likely paper only: best of whole-title and before-the-colon similarity."""
    full = difflib.SequenceMatcher(None, _norm_title(given), _norm_title(canonical)).ratio()
    head = difflib.SequenceMatcher(None, _norm_title(given.split(":")[0]),
                                   _norm_title(canonical.split(":")[0])).ratio()
    return max(full, head)


@tool
def check_citation(title: str, first_author: str | None = None, year: int | None = None) -> dict:
    """Verify a reference before repeating or correcting it. First finds the most likely catalogue
    paper (candidate), then checks each supplied detail against it. verdict: "verified" (title
    identical apart from case, punctuation, accents or an omitted subtitle, and the given author and
    year agree), "mismatch" (a likely paper exists but the title, first author or year differs:
    see mismatches and quote the candidate's canonical details), or "unverified" (no close title
    in the catalogue: say you could not verify it, do not call it fabricated). first_author may be
    a surname, "Ashish Vaswani", "Vaswani, A." or "Vaswani et al.". Use the candidate's arxiv_id
    with get_paper, format_citation or get_citation_count."""
    records = {**CATALOGUE, **_LIVE}
    best_score, best = max(((_candidate_score(title, p["title"]), p) for p in records.values()),
                           key=lambda s: s[0])
    if best_score < 0.85 and _live():
        try:
            for p in arxiv_client.search(title, 3):
                _LIVE[p["arxiv_id"]] = p
                if (s := _candidate_score(title, p["title"])) > best_score:
                    best_score, best = s, p
        except Exception:
            pass                                          # unreachable arXiv: fall through to unverified
    if best_score < 0.85:
        return {"verdict": "unverified", "title_similarity": round(best_score, 2),
                "closest_title": best["title"], "closest_arxiv_id": best["arxiv_id"],
                "note": "no close title in the catalogue; could not verify this reference"}

    match, similarity = _title_match(title, best["title"])
    mismatches = []
    if match == "different":
        mismatches.append({"field": "title", "given": title, "catalogue": best["title"],
                           "note": "a similar title is not the same title; check the candidate's details"})
    if first_author and not _author_matches(first_author, best["authors"][0]):
        mismatches.append({"field": "first_author", "given": first_author, "catalogue": best["authors"][0]})
    if year is not None and year != best["year"]:
        mismatches.append({"field": "year", "given": year, "catalogue": best["year"],
                           "note": "the catalogue year is the arXiv submission year, not the venue year"})
    candidate = {"arxiv_id": best["arxiv_id"], "title": best["title"], "first_author": best["authors"][0],
                 "year": best["year"], "url": best["url"]}
    return {"verdict": "mismatch" if mismatches else "verified", "candidate": candidate,
            "title_match": match, "title_similarity": round(similarity, 2), "mismatches": mismatches,
            # flat fields kept for callers of the earlier version
            "arxiv_id": best["arxiv_id"], "catalogue_title": best["title"],
            "catalogue_first_author": best["authors"][0], "catalogue_year": best["year"],
            "note": {"exact": "title matches the catalogue record",
                     "subtitle_omitted": "title matches the catalogue record with the subtitle omitted",
                     "different": "title differs from the candidate's; see mismatches"}[match]}


TOOLS = [search_papers, get_paper, format_citation, get_citation_count, check_citation]
