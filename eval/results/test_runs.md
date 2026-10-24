# Test runs

Historical runs: each tag was checked out into a temporary `git worktree`, the local `.env` copied
in, and `python -m pytest -q` run with the project venv (2026-10-24). Tool tests (Stage 4–5) need no
network; the other tests call `gpt-4.1-mini`.

| tag | commit | test files | result |
|---|---|---|---|
| stage-01 | 51d3e3a | test_stage01 (2 tests: graph has a `model` node; one invoke → 2 messages) | 2 passed |
| stage-02 | 1ab6b55 | test_stage01 (1), test_stage02 (1) | 2 passed |
| stage-03 | 16f77b2 | test_stage01 (1), test_stage02 (1), test_stage03 (2, parametrized) | 4 passed |
| stage-04 | ee29083 | + test_stage04 (3, offline tool tests) | 7 passed |
| stage-05 | e996a70 | + test_stage05 (8: loop, structure, 5 check_citation cases, stubbed count) | 15 passed |

Why the Stage 1 count drops from 2 to 1: `test_graph_compiles` asserted `"model" in graph.get_graph().nodes`;
Stage 2 replaced that node with `analyze → generate → review`, so the test was removed and only the
input/output contract (`test_single_node_answers`) was kept. `test_stage02` was later pointed at
`build_workflow()` when Stage 3 changed the entry point.

Current working tree (HEAD, 2026-10-24): `python -m pytest -q` → 15 passed.
