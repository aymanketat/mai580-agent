"""ONE state class. Add a field when a stage needs it. Never start a new class."""
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # Stage 1
    analysis: str      # Stage 2: what the student is really asking, and the facts needed
    draft: str         # Stage 2: first answer before review
