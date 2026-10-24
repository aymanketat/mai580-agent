# Grading evidence and how to reproduce it

## Historical evidence (preserved as run)

| file | what it is |
|---|---|
| `stage02_grading_prompt.js` | the prompt used for the Stage 1 vs Stage 2 quality grading, inside the Claude Code workflow script that ran it |
| `stage05_grading_prompt.js` | the prompt used for the Stage 4 vs Stage 5 task-success grading, inside its workflow script. **It states that both systems could call all five tools; Stage 4 had only three.** Kept unchanged as the historical record; see the corrected prompt below |
| `../results/stage02_grades.json` | three graders' outputs for Stage 1 vs Stage 2 |
| `../results/stage05_grades.json` | three graders' outputs for Stage 4 vs Stage 5 (under the incorrect tool-count premise) |
| `../results/stage05_regrade_q5_q8.json` | three graders' re-assessment of Q5, Q8 and Stage 4's Q2/Q3/Q9/Q10 with an explicit rubric and the correct tool lists |

The `.js` files are **Claude Code workflow scripts**, not standalone JavaScript: `agent(...)`,
`parallel(...)` and `meta` are provided by Claude Code's Workflow runtime, which spawned three
Claude instances (model `claude-fable-5-1`) per run and validated their JSON against the schema
in the script. Running them needs that runtime; the prompt text inside them can be used with any
capable chat model (see "Running with another model").

## Which files were system A and system B

| grading | system A | system B |
|---|---|---|
| Stage 1 vs Stage 2 | `eval/results/stage02.jsonl` (Stage 2 workflow) | `eval/results/stage01_on_stage02set.jsonl` (Stage 1 single node) |
| Stage 4 vs Stage 5 | `eval/results/stage05.jsonl` (Stage 5 loop, 5 tools) | `eval/results/stage04_on_multistep.jsonl` (Stage 4 one round, 3 tools) |

Graders saw only "A" and "B" (copies named `system_A.jsonl` / `system_B.jsonl`).

## Each stage's actual tool list

- Stage 4: `search_papers`, `get_paper`, `format_citation` — one tool round (`tools → final_answer`).
- Stage 5: the three above plus `get_citation_count` and `check_citation` — agent loop, `recursion_limit = 12`.

## Rubric (used for the re-assessment and the corrected report)

Three dimensions kept separate: **task completion** (which parts of the request were delivered),
**factual correctness** (no claim contradicts `data/papers.json` or is otherwise false; a legitimate
conference citation is not wrong because the catalogue holds the arXiv title), **tool provenance**
(citation strings and counts produced by the matching tool call). A part whose correctness depends on
tool results that were not logged is *unverified*. Score: 5 complete and correct with nothing
unsupported; 4 complete with a minor flaw (unsupported claim, flagged off-topic paper, wasted calls,
missing provenance); 3 main part delivered but a part missing or unverified; 2 a factual error or
most parts missing; 1 fabricated content. *Verified success* = all parts delivered, correct, and
traceable; *content-only success* = the same without the provenance requirement.

## Preparing the grading inputs

1. Copy the two result files to a scratch folder as `system_A.jsonl` and `system_B.jsonl` (see the
   mapping above; keep the mapping out of the grader's view).
2. Each line holds `text`, `label` (expected tool chain), `rationale`, `answer`, `tool_calls` (name
   and arguments, in order), `nodes`, `latency_s`, `tokens`, and, for Stage 5 runs only,
   `budget_exceeded` (whether the step budget stopped the run). **Tool results are not stored**, so
   live values such as citation counts can only be judged by whether a matching call exists.
3. Ground truth for authors, years, ids and titles is `data/papers.json`.

## Running with another model

Use `stage05_grading_prompt_corrected.md`: fill in the two file paths, give the model read access to
the three files (or paste them), and ask for the JSON object described at the end of the prompt.
Three independent runs with different lenses and a median per item reproduce the method used here.
