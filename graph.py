import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from memory_manager import add_memory, search_memory

load_dotenv()

llm = ChatOllama(
    model=os.getenv("LLM_MODEL", "llama3.1:8b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.7,
)

SYSTEM_PROMPT = """You are a helpful personal AI assistant with long-term memory.
Use the retrieved memories about the user to personalize your reply when relevant.
Do not mention that you are reading memories unless the user asks.

Memories about the user:
{memories}"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    memories: list[str]


def retrieve_memory(state: AgentState) -> dict:
    query = state["messages"][-1].content
    memories = search_memory(state["user_id"], query)
    return {"memories": memories}


def chat(state: AgentState) -> dict:
    memory_text = "\n".join(f"- {m}" for m in state["memories"]) or "No memories yet."
    system = SystemMessage(content=SYSTEM_PROMPT.format(memories=memory_text))
    response = llm.invoke([system, *state["messages"]])
    return {"messages": [response]}


def store_memory(state: AgentState) -> dict:
    user_msg = state["messages"][-2].content
    assistant_msg = state["messages"][-1].content
    add_memory(state["user_id"], user_msg, assistant_msg)
    return {}


builder = StateGraph(AgentState)
builder.add_node("retrieve_memory", retrieve_memory)
builder.add_node("chat", chat)
builder.add_node("store_memory", store_memory)

builder.add_edge(START, "retrieve_memory")
builder.add_edge("retrieve_memory", "chat")
builder.add_edge("chat", "store_memory")
builder.add_edge("store_memory", END)

agent = builder.compile()
