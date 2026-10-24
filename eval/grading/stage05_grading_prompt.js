export const meta = {
  name: 'grade-stage5-comparison',
  description: 'Three blind graders score task success of the one-round (Stage 4) vs looped (Stage 5) agent on 11 multi-step research requests',
  phases: [{ title: 'Grade', detail: 'three independent graders, blind to which system is which' }],
}

const SP = 'C:\\Users\\Wel\\AppData\\Local\\Temp\\claude\\C--Users-Wel-Documents-AeroVision\\3196c6be-8d1d-4cd0-b8d1-4bb11b01d634\\scratchpad\\grading5'
const PAPERS = 'C:\\Users\\Wel\\Documents\\mai580-agent\\data\\papers.json'

const GRADE_SCHEMA = {
  type: 'object',
  properties: {
    grades: { type: 'array', items: { type: 'object', properties: {
      q: { type: 'integer' }, system: { type: 'string' }, success: { type: 'boolean' }, score: { type: 'integer' },
      parts_done: { type: 'string' }, invented: { type: 'array', items: { type: 'string' } },
      unnecessary_calls: { type: 'array', items: { type: 'string' } }, comment: { type: 'string' } },
      required: ['q', 'system', 'success', 'score', 'parts_done', 'invented', 'unnecessary_calls', 'comment'] } },
    overall: { type: 'string' },
  },
  required: ['grades', 'overall'],
}

const LENSES = [
  'You are a strict fact-checker: any name, year, id or title that contradicts the catalogue caps the score at 2, and a citation string not traceable to a format_citation call is suspect.',
  'You are the graduate student who asked: did the answer complete EVERY part of the request, in a form you can paste into a thesis?',
  'You are an agent-systems reviewer: judge whether the tool-call sequence was sufficient and efficient for the request, and whether the answer honestly reports anything that could not be retrieved.',
]

const grades = await parallel(LENSES.map((lens, i) => () => agent(`
${lens}

Two anonymous research-assistant systems, A and B, answered the SAME 11 multi-step requests about machine-learning papers.
Both can call these tools: search_papers(query, max_results) -> cards (arxiv_id, title, first_author, author_count, year, snippet);
get_paper(arxiv_id) -> full record; format_citation(arxiv_id, style) -> APA/BibTeX/IEEE string built from the record;
get_citation_count(arxiv_id) -> Semantic Scholar count (live, so the NUMBER cannot be checked; accept any number attributed to Semantic
Scholar, but a number given without a get_citation_count call is invented); check_citation(title, first_author, year) -> verified /
mismatch / unverified plus arxiv_id.
Files (JSON Lines, same order in both):
  ${SP}\\system_A.jsonl
  ${SP}\\system_B.jsonl
Each object: text (request), label (the EXPECTED tool sequence written by the question author, e.g. "search_papers -> format_citation";
"single" means one call suffices), rationale, answer, tool_calls (the calls the system actually made, with args, in order), nodes,
latency_s, tokens. Ground truth for authors, author order, years, ids and titles is ${PAPERS}: read it and check every checkable claim.
Note: the catalogue year is the arXiv submission year (ReAct = 2022, GAN = 2014), and a reference list entry with "et al." is not valid APA 7.

For each (q = 1..11, system = "A" or "B") return: success (every part of the request answered correctly, with nothing invented;
honestly saying a count could not be retrieved counts as correct for that part only if no tool could have provided it), score 1-5,
parts_done (e.g. "2/3: found paper, cited it, did not give count"), invented (quote each false or untraceable claim; a BibTeX/APA
string produced without a format_citation call is 'untraceable' even if it looks right), unnecessary_calls (tool calls whose result the
request did not need or that repeat an earlier call), comment. Do not guess which system is which. Do not edit any file. Return data only.`,
  { label: `grader:${i + 1}`, phase: 'Grade', schema: GRADE_SCHEMA })))

return grades.filter(Boolean)
