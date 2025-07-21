# coding=utf-8
"""Tests for debate subgraph in CodeForge AI.

This module contains tests for multi-agent debate logic.
"""

from collections import deque
from unittest.mock import AsyncMock, patch

import pytest

from ..debate import run_debate, vote
from ..state import State


@pytest.mark.asyncio
async def test_debate_consensus() -> None:
    """Test debate with consensus vote; real-world: debate 'use async in Python?' expecting pro win."""
    state: State = {
        "task": "use async in Python?",
        "messages": [],
        "input": "",
        "task_queue": deque(),
        "private": {},
        "long_term": {},
    }
    with patch(
        "..debate.route_model",
        new_callable=AsyncMock,
        side_effect=lambda t, c: {
            "response": "Pro: Faster I/O"
            if "pro" in t
            else "Con: Complexity"
            if "con" in t
            else "Mod: Balance pro"
        },
    ):
        result: State = await run_debate(state)
        assert len(result["messages"]) == 6, (
            "Expected 3 messages per round for 2 rounds"
        )  # Insight: Full cycles
        assert vote(result), "Expected pro win on balance"
        assert "Faster I/O" in result["messages"][0]["content"], (
            "Expected real-world pro insight"
        )


@pytest.mark.asyncio
async def test_debate_refine() -> None:
    """Test debate with no consensus leading to refinement; real-world: 'microservices vs monolith' expecting con win/refine."""
    state: State = {
        "task": "microservices vs monolith",
        "messages": [],
        "input": "",
        "task_queue": deque(),
        "private": {},
        "long_term": {},
    }
    with patch(
        "..debate.route_model",
        new_callable=AsyncMock,
        side_effect=lambda t, c: {
            "response": "Pro: Scale"
            if "pro" in t
            else "Con: Overhead"
            if "con" in t
            else "Mod: Con wins"
        },
    ):
        result: State = await run_debate(state)
        assert "Refine based on debate" in result["task"], (
            "Expected refinement on no consensus"
        )  # Insight: Iteration path
        assert not vote(result), "Expected con majority"
        assert "Overhead" in result["messages"][1]["content"], (
            "Expected real-world con insight"
        )


@pytest.mark.asyncio
async def test_debate_single_round() -> None:
    """Test single round debate; real-world: simple 'REST vs GraphQL' for quick mod."""
    state: State = {
        "task": "REST vs GraphQL",
        "messages": [],
        "input": "",
        "task_queue": deque(),
        "private": {},
        "long_term": {},
    }
    with patch(
        "..debate.route_model",
        new_callable=AsyncMock,
        side_effect=lambda t, c: {
            "response": "Pro: Flexible"
            if "pro" in t
            else "Con: Overfetch"
            if "con" in t
            else "Mod: Pro"
        },
    ):
        result: State = await run_debate(state, rounds=1)
        assert len(result["messages"]) == 3, "Expected 3 messages for single round"
        assert vote(result), "Expected pro win"
        assert "Flexible" in result["messages"][0]["content"], (
            "Expected real-world pro insight"
        )
