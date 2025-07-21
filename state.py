# coding=utf-8
"""State definition for CodeForge AI LangGraph workflows.

This module defines the state structure and utilities for hierarchical memory.
"""

from collections import deque
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages


class State(TypedDict):
    """Workflow state for CodeForge AI.

    Attributes:
        messages: Short-term shared messages, annotated for addition.
        task_queue: In-memory task queue.
        private: Per-agent private state.
        long_term: Persistent long-term state via checkpointer.
        input: Input query or PRD.
    """

    messages: Annotated[List[Dict[str, Any]], add_messages]
    task_queue: deque[str]
    private: Dict[str, Any]
    long_term: Dict[str, Any]
    input: str


def cap_messages(state: State, max_messages: int = 50) -> State:
    """Cap shared messages to prevent bloat and ensure low latency.

    Args:
        state: Current workflow state.
        max_messages: Maximum messages to retain (default: 50).

    Returns:
        Updated state with capped messages.
    """
    if len(state["messages"]) > max_messages:
        state["messages"] = state["messages"][-max_messages:]
    return state


# Checkpointer for persistence (integrated in main.py)
checkpointer: MemorySaver = MemorySaver()
