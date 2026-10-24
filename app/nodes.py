"""Node functions. Each node returns an update (a dict), not the whole state."""
from langchain_core.messages import AIMessage, SystemMessage

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
