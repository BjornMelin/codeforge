# coding=utf-8
"""Integration tests for main orchestrator in CodeForge AI.

This module tests the full autonomy workflow.
"""

from collections import deque
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from ..config import settings
from ..main import run_autonomy_workflow


@pytest.mark.asyncio
async def test_autonomy_workflow_full() -> None:
    """Test end-to-end workflow with real-world PRD: 'Generate add function'; expect task pull, RAG fusion, debate refine, impl code."""
    with (
        patch(
            "..main.graphrag_plus",
            new_callable=AsyncMock,
            return_value=[{"content": "RAG: def add(a,b): return a+b"}],
        ) as mock_rag,
        patch(
            "..main.run_debate",
            new_callable=AsyncMock,
            return_value={"messages": ["Debate: Simple add ok"]},
        ) as mock_debate,
        patch(
            "..main.route_model",
            new_callable=AsyncMock,
            return_value={"response": "def add(a: int, b: int) -> int: return a + b"},
        ) as mock_route,
        patch.object(settings, "use_gpu", False),
    ):
        result: dict[str, Any] = await run_autonomy_workflow("Generate add function")
        assert "def add" in result.get("response", ""), (
            "Expected impl code from routing"
        )  # Insight: Full cycle output
        assert len(result.get("task_queue", deque())) == 0, (
            "Expected task pulled/processed"
        )
        assert len(result["messages"]) <= 50, "Expected capped messages"
        mock_rag.assert_called_once()  # Coverage: Research
        mock_debate.assert_called_once()  # Coverage: Debate
        mock_route.assert_called_once()  # Coverage: Implement


@pytest.mark.asyncio
async def test_autonomy_workflow_gpu() -> None:
    """Test workflow with GPU toggle; real-world: 'Optimize matrix mult' expecting compiled invoke."""
    with (
        patch(
            "..main.graphrag_plus",
            new_callable=AsyncMock,
            return_value=[{"content": "RAG: Use torch matmul"}],
        ) as mock_rag,
        patch(
            "..main.run_debate",
            new_callable=AsyncMock,
            return_value={"messages": ["Debate: GPU ok"]},
        ) as mock_debate,
        patch(
            "..main.route_model",
            new_callable=AsyncMock,
            return_value={"response": "torch.matmul(A, B)"},
        ) as mock_route,
        patch.object(settings, "use_gpu", True),
        patch(
            "torch.compile",
            return_value=AsyncMock(return_value={"response": "compiled result"}),
        ) as mock_compile,
    ):
        result: dict[str, Any] = await run_autonomy_workflow("Optimize matrix mult")
        assert "compiled result" in str(result), (
            "Expected GPU compiled invoke"
        )  # Insight: Perf boost path
        mock_compile.assert_called_once_with(
            mode="reduce-overhead"
        )  # Coverage: GPU toggle


@pytest.mark.asyncio
async def test_autonomy_workflow_no_task() -> None:
    """Test workflow without pub/sub message; real-world: empty queue fallback."""
    with (
        patch(
            "..main.graphrag_plus", new_callable=AsyncMock, return_value=[]
        ) as mock_rag,
        patch(
            "..main.run_debate", new_callable=AsyncMock, return_value={"messages": []}
        ) as mock_debate,
        patch(
            "..main.route_model", new_callable=AsyncMock, return_value={"response": ""}
        ) as mock_route,
        patch("redis.Redis.pubsub") as mock_pubsub,
    ):
        mock_pubsub.return_value.get_message.return_value = None
        result: dict[str, Any] = await run_autonomy_workflow("Empty task")
        assert len(result.get("task_queue", deque())) == 0, "Expected no task pulled"
        assert result.get("response", "") == "", (
            "Expected empty impl on no task"
        )  # Insight: Graceful fallback
