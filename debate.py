# coding=utf-8
"""Debate subgraph for CodeForge AI.

This module implements a multi-agent debate mechanism using LangGraph.
"""

from typing import Any, Dict

from langgraph.graph import StateGraph

from .config import settings
from .router import route_model
from .state import State


async def pro_agent(state: State) -> State:
    """Pro agent argues in favor of the task.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with pro argument.
    """
    pro: Dict[str, Any] = await route_model("Argue pro: " + state["task"], "reasoning")
    state["messages"].append({"role": "pro", "content": pro["response"]})
    return state


async def con_agent(state: State) -> State:
    """Con agent argues against the task.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with con argument.
    """
    con: Dict[str, Any] = await route_model("Argue con: " + state["task"], "reasoning")
    state["messages"].append({"role": "con", "content": con["response"]})
    return state


async def moderator_agent(state: State) -> State:
    """Moderator synthesizes arguments.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with moderation.
    """
    mod: Dict[str, Any] = await route_model(
        "Moderate: " + " ".join(m["content"] for m in state["messages"]), "reasoning"
    )
    state["messages"].append({"role": "moderator", "content": mod["response"]})
    return state


def vote(state: State) -> bool:
    """Perform simple majority vote for iteration.

    Args:
        state: Current workflow state.

    Returns:
        True if pros win (iterate), False otherwise.
    """
    pros: int = sum(1 for m in state["messages"] if "pro" in m["content"].lower())
    cons: int = len(state["messages"]) - pros
    return pros > cons


debate_subgraph: StateGraph = StateGraph(State)
debate_subgraph.add_node("pro", pro_agent)
debate_subgraph.add_node("con", con_agent)
debate_subgraph.add_node("moderator", moderator_agent)
debate_subgraph.add_parallel(["pro", "con"])
debate_subgraph.add_edge(["pro", "con"], "moderator")


async def run_debate(state: State, rounds: int = 2) -> State:
    """Run the debate subgraph for specified rounds.

    Args:
        state: Initial workflow state.
        rounds: Number of debate rounds (default: 2).

    Returns:
        Updated state after debate.
    """
    graph = debate_subgraph.compile(interrupt_before=["moderator"])
    for _ in range(rounds):
        state = await graph.ainvoke(state)
        if not vote(state):
            state["task"] += " Refine based on debate."
    return state
