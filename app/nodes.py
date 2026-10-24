"""Node functions. Each node returns an update (a dict), not the whole state."""
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import model
from app.state import State


# ---------------------------------------------------------------- Stage 1
def call_model(state: State) -> dict:
    reply = model.invoke(state["messages"])
    return {"messages": [reply]}


# ---------------------------------------------------------------- Stage 2
def analyze(state: State) -> dict:
    prompt = SystemMessage(
        "Restate the student's research question in one sentence and list the facts "
        "needed to answer it (paper titles, arXiv ids, authors, years, numbers)."
    )
    return {"analysis": model.invoke([prompt] + state["messages"]).content}


def generate(state: State) -> dict:
    prompt = SystemMessage(
        "You are a research assistant for machine-learning papers. "
        f"Request analysis:\n{state['analysis']}"
    )
    return {"draft": model.invoke([prompt] + state["messages"]).content}


def review(state: State) -> dict:
    prompt = SystemMessage(
        "Rewrite the draft so it is correct, polite, and under 120 words. "
        "If it states a fact you cannot verify (a citation, an author list, a number, a date), "
        "say so explicitly.\n\nDRAFT:\n" + state["draft"]
    )
    final = model.invoke([prompt]).content
    return {"messages": [AIMessage(final)]}


# ---------------------------------------------------------------- Stage 3
Intent = Literal["paper_lookup", "literature_search", "citation", "concept", "other"]

# v1 (90% on eval/routing.jsonl) described citation as "produce a BibTeX/APA citation, or
# verify/correct a reference"; both misses were reference checks routed to paper_lookup.
# v2 names the deciding cue: is the student working on a reference / bibliography entry?
INTENT_GUIDE = (
    "Classify the student's latest request. "
    "citation = the student is working on a reference or bibliography entry: wants a BibTeX/APA "
    "entry, or asks whether a reference they have is real, correct or complete (even if the "
    "question mentions authors or a year); "
    "paper_lookup = wants facts about ONE specific paper (authors, year, abstract, venue, what "
    "it claims) and is NOT checking or producing a reference; "
    "literature_search = find or list papers on a topic; "
    "concept = explain a method or idea without asking about a particular paper; "
    "other = anything else."
)


class Classification(BaseModel):
    intent: Intent = Field(description="The single best category for the request")


classifier = model.with_structured_output(Classification)


def classify(state: State) -> dict:
    result = classifier.invoke([SystemMessage(INTENT_GUIDE)] + state["messages"])
    return {"intent": result.intent}


def route_by_intent(state: State) -> str:
    """Routing function, not a node: returns the key the conditional edge looks up."""
    return state["intent"]


def _answer_node(system: str):
    """One answer node per intent: the same model call with a different system prompt."""
    def node(state: State) -> dict:
        return {"messages": [model.invoke([SystemMessage(system)] + state["messages"])]}
    return node


ASSISTANT = "You are a research assistant for machine-learning papers. "
answer_paper = _answer_node(
    ASSISTANT + "The student asks about ONE specific paper. Give its title, authors, year, arXiv id "
    "and a short summary. If you are not certain of a fact, say which."
)
answer_search = _answer_node(
    ASSISTANT + "The student wants papers on a topic. List 3-6 relevant papers with title, first "
    "author, year and one line on why each matters. Mark any entry you are not sure exists."
)
answer_citation = _answer_node(
    ASSISTANT + "Produce or check the requested citation. Use BibTeX unless another style is asked. "
    "Never invent a field: write ??? for anything you do not know."
)
answer_concept = _answer_node(
    ASSISTANT + "Explain the concept for a graduate student in under 200 words, with one concrete "
    "example and the canonical paper if you know it."
)
answer_general = _answer_node(
    ASSISTANT + "Answer briefly. If the request is outside papers and research, say what you can help with."
)
