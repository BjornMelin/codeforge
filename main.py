# coding=utf-8
"""Main orchestrator for CodeForge AI workflows.

This module sets up and runs the primary autonomy workflow using LangGraph.
"""

import os
from collections import deque
from typing import Any, Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from redis import Redis

from .config import settings
from .debate import run_debate
from .router import route_model
from .state import State, cap_messages
from .tools import graphrag_plus

redis: Redis = Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, dialect=3)
checkpointer: MemorySaver = MemorySaver()

workflow: StateGraph = StateGraph(State)
workflow.add_node("assign_task", lambda state: {"task_queue": deque([state["input"]])})
workflow.add_node(
    "research", lambda state: graphrag_plus(state["task_queue"].popleft())
)
workflow.add_node("debate", run_debate)
workflow.add_node("implement", lambda state: route_model(state["task"], "coding"))
workflow.add_edge("assign_task", "research")
workflow.add_edge("research", "debate")
workflow.add_edge("debate", "implement")
workflow.add_edge("implement", END)

graph = workflow.compile(checkpointer=checkpointer)


async def run_autonomy_workflow(input: str) -> Dict[str, Any]:
    """Run the full autonomy workflow from input.

    Args:
        input: Initial input query or PRD.

    Returns:
        Final workflow result dictionary.
    """
    state: State = {
        "input": input,
        "task_queue": deque(),
        "messages": [],
        "private": {},
        "long_term": {},
    }
    redis.publish("tasks", input)
    sub = redis.pubsub()
    sub.subscribe("tasks")
    message = sub.get_message(timeout=1)
    if message:
        state["task_queue"].append(message["data"].decode())  # type: ignore

    if settings.use_gpu:
        import torch

        @torch.compile(mode="reduce-overhead")
        async def compiled_invoke(state: State) -> Dict[str, Any]:
            return await graph.ainvoke(state)

        result: Dict[str, Any] = await compiled_invoke(state)
    else:
        result = await graph.ainvoke(state)

    result = cap_messages(result)
    checkpointer.save(result)
    return result


# Sample usage
sample_prd: str = "Generate a simple Python function to add two numbers."
