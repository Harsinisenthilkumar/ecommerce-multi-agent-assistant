from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    current_intent: NotRequired[str]
    retrieved_memories: NotRequired[list[str]]