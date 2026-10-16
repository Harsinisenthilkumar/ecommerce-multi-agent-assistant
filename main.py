import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from agents.graph import app_graph

load_dotenv()
CONSOLE_THREAD_ID = "ecommerce-console"

async def run_console_cli():
    print("=" * 50)
    print("Multi-Agent E-Commerce Assistant (Gemini Flash + LangGraph)")
    print("Type your message below (or type 'exit' to quit):")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting assistant. Goodbye!")
                break
            
            if not user_input.strip():
                continue

            # Route input through your LangGraph multi-agent flow
            state_input = {"messages": [HumanMessage(content=user_input)]}
            result = await app_graph.ainvoke(
                state_input,
                config={"configurable": {"thread_id": CONSOLE_THREAD_ID}},
            )
            
            # Find the first non-empty assistant response from the end
            final_message = "No response generated."
            if "messages" in result:
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                        final_message = msg.content
                        break
            
            print(f"Agent: {final_message}")
            
        except KeyboardInterrupt:
            print("\nExiting assistant. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_console_cli())