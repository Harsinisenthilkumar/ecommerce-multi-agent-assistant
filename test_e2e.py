import asyncio

from langchain_core.messages import HumanMessage
from agents.graph import app_graph


async def run_end_to_end_test():
    test_queries = [
        "Can you check if SKU-101 is in stock?",
        "Please order 2 units of SKU-101 and deliver them to 123 Main Street, Springfield.",
        "Can you check the status of my order ORD-88492?"
    ]

    print("="*60 + "\nSTARTING MULTI-AGENT WORKFLOW SIMULATION TEST\n" + "="*60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Step {i} ---")
        print(f"User: {query}")
        
        result = await app_graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": "ecommerce-e2e"}},
        )
        
        for msg in result["messages"]:
            if msg.type == "ai" and msg.content:
                print(f"Agent: {msg.content}")
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    print(f"🛠️ [Tool Invoked]: {call['name']} | Args: {call['args']}")

if __name__ == "__main__":
    asyncio.run(run_end_to_end_test())