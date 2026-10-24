# Corrected grading prompt — Stage 4 vs Stage 5 task success

This is the prompt of `stage05_grading_prompt.js` with the tool lists corrected and repository-relative
paths. It is model-agnostic. Replace `<scratch>` with the folder holding the blinded copies.

---

You are grading saved outputs of two research-assistant systems, A and B, on the same 11 requests
about machine-learning papers. Files (JSON Lines, same order in both; q = line number 1..11):

- `<scratch>/system_A.jsonl`
- `<scratch>/system_B.jsonl`
- `data/papers.json` — ground truth for titles, authors (in order), years, arXiv ids, abstracts.

Each object has: `text` (request), `label` (the expected tool chain written by the question author;
"single" means one call suffices), `rationale`, `answer`, `tool_calls` (name and arguments, in
order), `nodes`, `latency_s`, `tokens`. Tool *results* are not stored.

Tool lists — read them carefully, they differ:
- One system has only `search_papers(query, max_results)`, `get_paper(arxiv_id)` and
  `format_citation(arxiv_id, style)`, and makes exactly one round of tool calls.
- The other system has those three plus `get_citation_count(arxiv_id)` (live Semantic Scholar count)
  and `check_citation(title, first_author, year)` (verified / mismatch / unverified plus arxiv_id),
  and can loop. You can tell which is which from `tool_calls`; do not penalise a system for not
  calling a tool it does not have, and treat an honest "could not retrieve" as correct for that part.

Rules:
- Task completion: list the parts of the request and which were delivered (`parts_done`).
- Factual correctness (`factual_errors`): quote claims that contradict `data/papers.json` or are
  demonstrably false. A legitimate alternative citation (for example a conference version with its
  own title, venue and pages) is not an error merely because the catalogue holds the arXiv record;
  mark it `unverified_externally` if you cannot check it.
- `unsupported_claims`: stated as fact, backed by no tool result, not checkable in the catalogue.
- `provenance_gaps`: citation strings or counts not produced by the matching tool call, even if correct.
- Relevance: for every paper presented as matching the request, judge from its abstract whether it is
  on-topic ("yes" / "borderline" / "no") and whether the system flagged it.
- `success` = every part delivered, every paper presented as relevant on-topic or flagged, no factual
  error, and citations/counts traceable. Also report `content_only_success` (same without provenance).
- `score` 1–5 as in `eval/grading/README.md`; `confidence` high/medium/low with a one-line
  `uncertainty_note`.

Return one JSON object: `{"grades": [{"q", "system", "success", "content_only_success", "score",
"parts_done", "relevance", "factual_errors", "unsupported_claims", "provenance_gaps", "confidence",
"uncertainty_note", "comment"} ...], "overall": "..."}` for q = 1..11 and both systems.
Do not guess which system is which. Return data only.
