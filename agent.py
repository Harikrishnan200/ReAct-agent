"""
ReAct Agent using LangGraph + Ollama (Llama) + Wikipedia Tool
--------------------------------------------------------------
This script creates a ReAct (Reasoning + Acting) agent that:
  - Uses a locally running Ollama Llama model as the LLM
  - Has access to a Wikipedia search tool
  - Decides on its own whether to call the tool or answer directly
  - Prints the full reasoning chain and the final answer

Requirements:
  - Ollama must be installed and running locally
  - Llama model must be pulled: `ollama pull llama3.2`

"""


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


# ─────────────────────────────────────────────────────────────
# STEP 1: CONFIGURE THE WIKIPEDIA TOOL
# top_k_results   — how many Wikipedia results to fetch
# doc_content_chars_max — how many characters to return per result
# ─────────────────────────────────────────────────────────────
def build_wikipedia_tool() -> WikipediaQueryRun:
    api_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=300
    )
    return WikipediaQueryRun(api_wrapper=api_wrapper)


# ─────────────────────────────────────────────────────────────
# STEP 2: CONFIGURE THE LOCAL OLLAMA LLM
# model      — the Ollama model name you have pulled locally
# temperature — 0 = deterministic/factual responses
# base_url   — default Ollama server address
# ─────────────────────────────────────────────────────────────
def build_llm() -> ChatOllama:
    return ChatOllama(
        model="llama3",          # change to "llama3.1" or "llama.2" if needed
        temperature=0,
        base_url="http://localhost:11434"
    )


# ─────────────────────────────────────────────────────────────
# STEP 3: BUILD THE REACT AGENT
# create_react_agent wires the LLM and tools together into a
# LangGraph state machine that loops until it has a final answer:
#   Think → (optionally) Call Tool → Observe result → Think again → Answer
# ─────────────────────────────────────────────────────────────
def build_agent(llm: ChatOllama, tools: list):
    return create_react_agent(llm, tools)


# ─────────────────────────────────────────────────────────────
# STEP 4: RUN A QUERY THROUGH THE AGENT
# Prints each message in the chain so you can see:
#   - The original human question
#   - The AI's decision to call a tool (if any)
#   - The tool's response
#   - The AI's final synthesised answer
# ─────────────────────────────────────────────────────────────
def run_query(agent_executor, query: str) -> str:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    response = agent_executor.invoke({
        "messages": [HumanMessage(content=query)]
    })

    # Print the full reasoning chain
    for i, msg in enumerate(response["messages"]):
        msg_type = msg.__class__.__name__
        content = msg.content if msg.content else "[tool call triggered — no text content]"
        print(f"\n[Step {i + 1}] {msg_type}:\n{content}")

    # Return only the final answer
    final_answer = response["messages"][-1].content
    return final_answer


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # Build components
    wiki_tool = build_wikipedia_tool()
    llm = build_llm()
    tools = [wiki_tool]
    agent_executor = build_agent(llm, tools)

    # ── Test 1: Simple greeting — no tool needed ──
    # Agent should answer directly without calling Wikipedia
    answer1 = run_query(agent_executor, "hi!")
    print(f"\nFINAL ANSWER:\n{answer1}")

    # ── Test 2: Factual question — tool will be called ──
    # Agent should decide to call Wikipedia and use the result
    answer2 = run_query(agent_executor, "what is agentic ai")
    print(f"\nFINAL ANSWER:\n{answer2}")

    # ── Test 3: Another factual question ──
    answer3 = run_query(agent_executor, "who is Alan Turing")
    print(f"\nFINAL ANSWER:\n{answer3}")


if __name__ == "__main__":
    main()