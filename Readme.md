# ReAct Agent — LangGraph + Ollama (Llama) + Wikipedia

A simple AI agent that uses the **ReAct (Reasoning + Acting)** pattern to answer questions.
It decides on its own whether to search Wikipedia or answer directly from its knowledge.

---

## How It Works

```
User Question
     ↓
  Agent Thinks
     ↓
Needs external info?
   ├── YES → Calls Wikipedia Tool → Gets result → Thinks again → Answers
   └── NO  → Answers directly
```

---

## Project Structure

```
├── agent.py           # Main agent script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed on your machine

---

## Setup

**1. Install Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**2. Pull the Llama model**
```bash
ollama pull llama3.2
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

---

## Run

Make sure Ollama is running first:
```bash
ollama serve
```

Then run the agent:
```bash
python agent.py
```

---

## Example Output

```
============================================================
QUERY: what is agentic ai
============================================================

[Step 1] HumanMessage:
what is agentic ai

[Step 2] AIMessage:
[tool call triggered — no text content]

[Step 3] ToolMessage:
Page: Intelligent agent
Summary: In artificial intelligence, an intelligent agent is an entity
that perceives its environment, takes actions autonomously to achieve goals...

[Step 4] AIMessage:
Agentic AI refers to intelligent agents that can perceive their environment,
take autonomous actions to achieve specific goals, and improve their
performance through learning or acquiring knowledge.

FINAL ANSWER:
Agentic AI refers to intelligent agents...
```

---

## Configuration

To change the Llama model, edit the `build_llm()` function in `agent.py`:

```python
ChatOllama(
    model="llama3.2",   # change to "llama3.1" or "llama3" if needed
    temperature=0,
    base_url="http://localhost:11434"
)
```

Available models (must be pulled first with `ollama pull <model>`):

| Model | Size | Speed |
|---|---|---|
| `llama3.2` | 3B | Fastest |
| `llama3.1` | 8B | Balanced |
| `llama3`   | 8B | Older version |

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core framework |
| `langchain-community` | Wikipedia tool integration |
| `langchain-ollama` | Ollama LLM integration |
| `langgraph` | ReAct agent state machine |
| `wikipedia` | Wikipedia API client |