"""
agent_graph.py

The actual LangGraph graph (Task 4 + Task 5). 5 nodes:
    guardrail -> intent -> [retrieve | tool_call] -> generate

One conditional edge (out of "intent") makes the graph actually branch by
intent instead of always running every node. Short-term conversational state
(specifically: the last order ID mentioned) is carried across turns using
LangGraph's own MemorySaver checkpointer, keyed by thread_id -- reusing the
same thread_id across invoke() calls carries state forward; a fresh
thread_id starts with that state correctly absent. See
transcripts/07_multiturn.md and transcripts/08_fresh_conversation.md.
"""

from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from guardrails import check_prompt_injection, check_groundedness
from retriever import retrieve_chunks, dedupe_to_documents
from tools import check_return_risk, classify_product_image
from mock_llm import (
    classify_intent, extract_order_id, extract_image_path,
    compose_policy_answer, compose_refusal_answer,
    compose_return_risk_answer, compose_image_classification_answer,
    compose_injection_blocked_answer,
)


class AgentState(TypedDict, total=False):
    user_input: str
    order_features: Optional[dict]
    image_path: Optional[str]
    intent: str
    blocked: bool
    block_reason: Optional[str]
    retrieved_docs: List[dict]
    tool_result: Optional[dict]
    last_order_id: Optional[str]
    final_answer: dict


# --- nodes -------------------------------------------------------------

def guardrail_node(state: AgentState) -> dict:
    check = check_prompt_injection(state["user_input"])
    return {"blocked": check["blocked"], "block_reason": check["reason"]}


def intent_node(state: AgentState) -> dict:
    if state.get("blocked"):
        return {"intent": "blocked"}

    intent = classify_intent(state["user_input"])
    found_order_id = extract_order_id(state["user_input"])
    # carry the last order id forward if this turn didn't mention a new one
    last_order_id = found_order_id or state.get("last_order_id")
    return {"intent": intent, "last_order_id": last_order_id}


def route_after_intent(state: AgentState) -> str:
    if state.get("blocked"):
        return "generate"
    if state["intent"] == "policy":
        return "retrieve"
    return "tool_call"


def retrieve_node(state: AgentState) -> dict:
    chunks = retrieve_chunks(state["user_input"], k=3)
    docs = dedupe_to_documents(chunks)
    return {"retrieved_docs": docs}


def tool_call_node(state: AgentState) -> dict:
    if state["intent"] == "return_risk":
        order_features = state.get("order_features")
        if order_features is None:
            return {"tool_result": None}
        result = check_return_risk(order_features)
        return {"tool_result": result}

    if state["intent"] == "product_category":
        image_path = state.get("image_path") or extract_image_path(state["user_input"])
        result = classify_product_image(image_path)
        return {"tool_result": result}

    return {}


def generate_node(state: AgentState) -> dict:
    if state.get("blocked"):
        answer = compose_injection_blocked_answer(state["block_reason"])
    elif state["intent"] == "policy":
        docs = state.get("retrieved_docs", [])
        grounded = check_groundedness(docs)
        if grounded["grounded"]:
            answer = compose_policy_answer(docs)
        else:
            answer = compose_refusal_answer(grounded["top_score"], grounded["threshold"])
    elif state["intent"] == "return_risk":
        if state.get("tool_result"):
            answer = compose_return_risk_answer(state["tool_result"], state.get("last_order_id"))
        else:
            answer = {
                "answer": "I need this order's features (category, price, payment "
                           "method, etc.) to assess return risk -- none were provided.",
                "source": "return_risk_tool",
                "confidence": 0.0,
            }
    elif state["intent"] == "product_category":
        answer = compose_image_classification_answer(state["tool_result"])
    else:
        answer = {"answer": "I couldn't tell what you're asking -- could you rephrase?",
                   "source": "policy_kb", "confidence": 0.0}
    return {"final_answer": answer}


# --- graph assembly ------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("intent", intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("tool_call", tool_call_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("guardrail")
    graph.add_edge("guardrail", "intent")
    graph.add_conditional_edges(
        "intent", route_after_intent,
        {"retrieve": "retrieve", "tool_call": "tool_call", "generate": "generate"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("tool_call", "generate")
    graph.add_edge("generate", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def invoke_agent(user_input: str, thread_id: str = "default",
                  order_features: dict = None, image_path: str = None) -> dict:
    """One conversational turn. Re-use the same thread_id across calls to
    carry state (e.g. the last order id) forward; use a new thread_id to
    start a fresh conversation with that state absent."""
    compiled = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {"user_input": user_input}
    if order_features is not None:
        input_state["order_features"] = order_features
    if image_path is not None:
        input_state["image_path"] = image_path
    return compiled.invoke(input_state, config=config)
