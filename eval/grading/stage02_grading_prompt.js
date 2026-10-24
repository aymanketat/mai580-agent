export const meta = {
  name: 'grade-stage2-comparison',
  description: 'Three blind graders score Stage 1 vs Stage 2 answers on the same 10 research questions against the paper catalogue',
  phases: [{ title: 'Grade', detail: 'three independent graders, blind to which system is which' }],
}

const SP = 'C:\\Users\\Wel\\AppData\\Local\\Temp\\claude\\C--Users-Wel-Documents-AeroVision\\3196c6be-8d1d-4cd0-b8d1-4bb11b01d634\\scratchpad\\grading'
const PAPERS = 'C:\\Users\\Wel\\Documents\\mai580-agent\\data\\papers.json'

const GRADE_SCHEMA = {
  type: 'object',
  properties: {
    grades: { type: 'array', items: { type: 'object', properties: {
      q: { type: 'integer' }, system: { type: 'string' }, score: { type: 'integer' },
      correct: { type: 'boolean' }, invented: { type: 'array', items: { type: 'string' } },
      disclaimer: { type: 'boolean' }, words: { type: 'integer' }, comment: { type: 'string' } },
      required: ['q', 'system', 'score', 'correct', 'invented', 'disclaimer', 'words', 'comment'] } },
    overall: { type: 'string' },
  },
  required: ['grades', 'overall'],
}

const LENSES = [
  'You are a strict fact-checker: a single wrong name, number, year or id caps the score at 2.',
  'You are the graduate student who asked: score how useful and well-sized the answer is for a thesis, given it must also be correct.',
  'You are a careful journal reviewer: weigh correctness first, then whether the answer honestly marks what it cannot verify.',
]

const grades = await parallel(LENSES.map((lens, i) => () => agent(`
${lens}

Two anonymous systems, A and B, answered the SAME 10 machine-learning research questions. Grade all 20 answers.
Files (JSON Lines, one object per question, in the same order in both files):
  ${SP}\\system_A.jsonl
  ${SP}\\system_B.jsonl
Each object has: text (the question), label (short | long | needs_data), rationale (what the question-writer expected), answer.
Ground truth for every paper fact (authors, author counts, years, arXiv ids, versions, titles) is ${PAPERS}: read it and check
every checkable claim against it. Facts not in the file (e.g. live citation counts) are unknowable; the right answer is to say so.

For each (q = 1..10, system = "A" or "B") return: score 1-5 (5 = fully correct, well-sized, nothing invented; 3 = mostly right
with a notable flaw; 1 = wrong or fabricated), correct (no false factual claim), invented (list each false or unverifiable-but-asserted
claim, quoting it), disclaimer (does the answer explicitly say it cannot verify something), words (approximate word count), comment.
Do not guess which system is which and do not reward or punish length for its own sake. Do not edit any file. Return data only.`,
  { label: `grader:${i + 1}`, phase: 'Grade', schema: GRADE_SCHEMA })))

return grades.filter(Boolean)
