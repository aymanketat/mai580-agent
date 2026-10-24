# mai580-agent — Research Assistant

MAI 580 course project (Use Case B). A LangGraph agent that answers questions about
machine-learning papers: it finds papers in a local catalogue, returns records and citations,
checks references a student quotes, and reports Semantic Scholar citation counts. The agent is
grown stage by stage following the course tutorial; **Stages 1–5 are complete** (tags `stage-01`
to `stage-05`). Stages 6–16 are not started: `app/guards.py`, `app/monitor.py`, `app/planning.py`,
`app/rag.py` and `app/reflection.py` are empty placeholders for them.

## What is implemented (Stages 1–5)

| stage | graph | what it adds |
|---|---|---|
| 1 | `START → model → END` | one `State` (`messages` with the `add_messages` reducer) |
| 2 | `analyze → generate → review` | focused prompts sharing data through state fields (kept as `build_workflow()`) |
| 3 | `classify` → answer nodes | structured-output router over `paper_lookup / literature_search / citation / concept / other` |
| 4 | router + `agent → tools → final_answer` | tools `search_papers`, `get_paper`, `format_citation` (one tool round) |
| 5 | router + `agent ⇄ tools` | the ReAct loop with `recursion_limit = 12`; tools `get_citation_count`, `check_citation` |

Layout: `app/` (agent code: `llm.py`, `state.py`, `nodes.py`, `graph.py`, `tools.py`,
`arxiv_client.py`), `data/` (catalogue, snapshot, runtime cache), `tests/` (one file per stage plus
regression tests), `eval/` (question sets, raw runs, grades, grader prompts), `scripts/`
(experiment harness), `notes/` (milestone notes and PDF), `tutorial/` (the course tutorial).

## Setup (Windows)

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # then add your OPENAI_API_KEY
```

If `pip` fails with a certificate error behind corporate or antivirus HTTPS inspection, point it
at a CA bundle that includes that root certificate (`pip config set global.cert <bundle.pem>`) and
set `SSL_CERT_FILE` to the same bundle in `.env` so the OpenAI client trusts it too.

## Running

```
python run.py                                      # chat loop (type exit to quit)
python scripts/inspect_graph.py                    # print the graph as Mermaid
python scripts/inspect_graph.py "your question"    # stream one run node by node
```

The chat loop, the inspection script and the experiment runner all apply the step budget from
`app/graph.py` (`RECURSION_LIMIT`) and show the fallback message when it is reached. All three
import the model at start-up (`app/llm.py`), so `OPENAI_API_KEY` must be set even for the
diagram-only command, which makes no model call.

## Tests

```
pytest -m "not live"      # offline: tools, cache, retry, reference checks, graph structure (no key needed)
pytest -m live            # calls the model (the Stage 5 loop test may also call Semantic Scholar)
pytest                    # both; live tests are skipped automatically when no real key is set
```

Offline tests never read or write `data/cache/` or the snapshot (a temporary directory is used)
and do not depend on the `ARXIV_LIVE` / `S2_LIVE` values in `.env`.

## Data sources and configuration

- `data/papers.json` — 23 real arXiv records, built by `python scripts/build_catalogue.py` from the
  arXiv API. The catalogue is the agent's private knowledge; tools read it offline.
- Semantic Scholar — `get_citation_count` queries the public Graph API (unauthenticated, shared
  rate limit). Successful responses are cached in `data/cache/semantic_scholar/` (gitignored) with
  their retrieval time and reused for `S2_CACHE_TTL_HOURS`; after that a fresh lookup is attempted
  and, if Semantic Scholar does not answer, the old value is returned marked `stale` with its
  original retrieval time. `data/citations_snapshot.json` is a committed, dated copy of real
  responses (`python scripts/refresh_citations.py`; it currently covers part of the catalogue and
  keeps earlier entries when a refresh is rate-limited) used only when no cache entry exists; its
  values are labelled `source: "snapshot"`.
- Configuration (`.env`, see `.env.example`): `ARXIV_LIVE` (default off) lets `search_papers`,
  `get_paper`, `format_citation` and `check_citation` query arXiv for papers outside the catalogue;
  `S2_LIVE` (default **on**; set `0` to turn Semantic Scholar off) and `S2_CACHE_TTL_HOURS`. With
  `ARXIV_LIVE=0` and `S2_LIVE=0` the tools run entirely from `data/`. Model calls (`run.py`, the
  agent nodes, `live` tests) always need `OPENAI_API_KEY`.

## Reports and evidence

- `notes/M1.md` — Milestone 1 progress note; `notes/M1_Progress_Ayman_Ketat.pdf` — its two-page export.
- `notes/M1_detailed.md` — full report: every experiment table, reflection answers, corrections.
- `eval/` — question sets (`*.jsonl`), raw runs and grades (`results/`), grader prompts and how to
  reproduce the grading (`grading/README.md`).

## Known limitations

- `search_papers` is lexical: off-topic hits for conceptual queries (Stage 8 adds retrieval).
- No memory between turns: each question starts from an empty state (Stage 6).
- The experiment runner logs tool calls but not tool results, so some historical answers (for
  example citation counts) cannot be audited after the fact (Stage 12 adds run logging).
- Semantic Scholar rate-limits readily; counts are reported with their retrieval date and may be
  cached or stale. Counts are per Semantic Scholar *record*: the record for an arXiv id can differ
  from the record for the published version of the same paper (arXiv 1406.2661, the GAN preprint,
  reports 169 citations while the NeurIPS version is a separate record).
- The Stage 4 → Stage 5 comparison changes the loop and the tool set together; it does not isolate
  the loop's effect.
- Routing accuracy (100% on 20 requests) was tuned on the same requests it was measured on.
