from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import os
import sys

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not OPENAI_API_KEY:
    print("Missing OPENAI_API_KEY in environment.", file=sys.stderr)
    sys.exit(1)
if not FIRECRAWL_API_KEY:
    print("Warning: FIRECRAWL_API_KEY not set. Firecrawl tools may fail.", file=sys.stderr)

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

server_params = StdioServerParameters(
    command="npx",
    args=["firecrawl-mcp"],
    env={"FIRECRAWL_API_KEY": FIRECRAWL_API_KEY} if FIRECRAWL_API_KEY else None,
)

def extract_text(response):
    """
    Tries a few likely shapes to get a printable assistant reply.
    """
    try:
        # LangGraph prebuilt often returns a dict with 'messages'
        msgs = response.get("messages")
        if msgs:
            last = msgs[-1]
            # last may be a dict or a LangChain message obj
            if isinstance(last, dict):
                return last.get("content") or str(last)
            content = getattr(last, "content", None)
            if content:
                return content
    except Exception:
        pass
    # Fallbacks
    for key in ("output", "response", "final", "final_response"):
        if isinstance(response, dict) and response.get(key):
            return response[key]
    return str(response)

async def main():
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                agent = create_react_agent(model, tools)

                print("Available Tools:", ", ".join(getattr(t, "name", "tool") for t in tools))
                print("-" * 60)

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You can scrape websites, crawl pages, and extract data using Firecrawl tools. "
                            "Think step by step and use the appropriate tools."
                        ),
                    }
                ]

                while True:
                    try:
                        user_input = input("\nYou: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\nGoodbye")
                        break

                    if user_input.lower() in {"quit", "exit"}:
                        print("Goodbye")
                        break
                    if not user_input:
                        continue

                    # keep history but avoid runaway growth
                    messages.append({"role": "user", "content": user_input[:175000]})
                    # Optional: cap history length
                    if len(messages) > 40:
                        # keep system + last ~20 exchanges
                        messages = [messages[0]] + messages[-39:]

                    try:
                        result = await agent.ainvoke({"messages": messages})
                        text = extract_text(result)
                        print("\nAgent:", text)
                        # append assistant message back to history (helps some agents)
                        messages.append({"role": "assistant", "content": text})
                    except Exception as e:
                        print("Error during agent invoke:", e, file=sys.stderr)
    except FileNotFoundError as e:
        # npx not found or firecrawl-mcp not resolvable
        print("Launch error:", e, file=sys.stderr)
        print("Make sure Node+npx are installed and `npx firecrawl-mcp --help` works.", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
