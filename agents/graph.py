from functools import lru_cache
from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI  # pyright: ignore[reportMissingImports]
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from config import Config
from agents.state import AgentState
from database import check_order_status, check_stock, create_order

inventory_tools = [check_stock]
order_tools = [create_order]
tracking_tools = [check_order_status]
all_tools = [*inventory_tools, *order_tools, *tracking_tools]

ROUTE_PROMPT = """Classify the user's latest request into exactly one route:
inventory for stock or availability questions, orders for placing or changing
an order, tracking for delivery or order-status questions, and general for
anything else. Do not answer the user; return only the structured route."""

AGENT_PROMPTS = {
    "inventory": "You handle product inventory questions. Use your inventory tool to check the mock database.",
    "orders": "You handle order creation. Use your order tool to create orders in the mock database. Ask for a SKU, positive quantity, and delivery address if any are missing. Never claim success unless the tool confirms it.",
    "tracking": "You handle order tracking. Use your tracking tool to look up the mock database. Ask for an order ID if it is missing.",
    "general": "You are a concise e-commerce assistant. Answer general questions briefly.",
}


class RouteDecision(BaseModel):
    intent: Literal["inventory", "orders", "tracking", "general"]


@lru_cache(maxsize=1)
def _get_base_llm() -> ChatGoogleGenerativeAI:
    if not Config.GOOGLE_API_KEY:
        raise RuntimeError(
            "Gemini credentials are missing. Set GOOGLE_API_KEY or GEMINI_API_KEY."
        )

    return ChatGoogleGenerativeAI(
        model=Config.GEMINI_MODEL,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def _get_router_llm():
    return _get_base_llm().with_structured_output(RouteDecision)


async def supervisor_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, str]:
    """Classify the latest request and select a specialized agent."""
    if not state.get("messages"):
        raise ValueError("At least one user or conversation message is required.")

    decision = await _get_router_llm().ainvoke(
        [SystemMessage(content=ROUTE_PROMPT), *state["messages"]],
        config=config,
    )
    return {"current_intent": decision.intent}


async def _run_agent(
    state: AgentState,
    config: RunnableConfig,
    intent: str,
    tools: list[BaseTool],
) -> dict[str, object]:
    messages = [SystemMessage(content=AGENT_PROMPTS[intent]), *state["messages"]]
    model = _get_base_llm()
    if tools:
        model = model.bind_tools(tools)
    response = await model.ainvoke(messages, config=config)
    return {"messages": [response], "current_intent": intent}


async def inventory_agent(
    state: AgentState, config: RunnableConfig
) -> dict[str, object]:
    return await _run_agent(state, config, "inventory", inventory_tools)


async def order_agent(
    state: AgentState, config: RunnableConfig
) -> dict[str, object]:
    return await _run_agent(state, config, "orders", order_tools)


async def tracking_agent(
    state: AgentState, config: RunnableConfig
) -> dict[str, object]:
    return await _run_agent(state, config, "tracking", tracking_tools)


async def general_agent(
    state: AgentState, config: RunnableConfig
) -> dict[str, object]:
    return await _run_agent(state, config, "general", [])


agent_nodes = {
    "inventory": inventory_agent,
    "orders": order_agent,
    "tracking": tracking_agent,
    "general": general_agent,
}


def route_from_supervisor(state: AgentState) -> str:
    intent = state.get("current_intent", "general")
    return intent if intent in agent_nodes else "general"


def route_agent_result(state: AgentState) -> str:
    """Run requested mock database tools before returning the response."""
    if not state.get("messages"):
        raise ValueError("An agent cannot route an empty conversation.")
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END


def route_after_tools(state: AgentState) -> str:
    intent = state.get("current_intent", "general")
    return intent if intent in agent_nodes else "general"


workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("tools", ToolNode(all_tools))
for agent_name, agent_node in agent_nodes.items():
    workflow.add_node(agent_name, agent_node)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {agent_name: agent_name for agent_name in agent_nodes},
)
for agent_name in agent_nodes:
    workflow.add_conditional_edges(
        agent_name,
        route_agent_result,
        {"tools": "tools", END: END},
    )
workflow.add_conditional_edges(
    "tools",
    route_after_tools,
    {agent_name: agent_name for agent_name in agent_nodes},
)

# MemorySaver retains conversation state in this process. Callers must pass a
# stable configurable.thread_id for each conversation.
memory_checkpointer = MemorySaver()
app_graph = workflow.compile(checkpointer=memory_checkpointer)