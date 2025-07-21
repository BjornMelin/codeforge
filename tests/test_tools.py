# coding=utf-8
"""Tests for retrieval tools in CodeForge AI.

This module contains unit and integration tests for GraphRAG+ functionality.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from ..config import settings
from ..tools import graphrag_plus


@pytest.mark.asyncio
async def test_graphrag_plus_web_trigger() -> None:
    """Test GraphRAG+ with web fallback when vector results empty;
    real-world: query for 'python async best practices'."""
    with (
        patch(
            "..tools.qdrant.aquery", new_callable=AsyncMock, return_value=[]
        ) as mock_qdrant,
        patch(
            "..tools.tavily.search",
            return_value=[
                {
                    "content": "Async best practices: use asyncio.gather for concurrency."
                }
            ],
        ) as mock_tavily,
        patch(
            "..tools.neo4j_driver.session", new_callable=AsyncMock
        ) as mock_session,
        patch.object(settings, "use_async", True),
        patch.object(settings, "use_sparse", False),
    ):
        mock_session.return_value.run = AsyncMock(
            return_value=[{"n": {"content": "Graph node on async."}}]
        )
        results: list[dict[str, Any]] = await graphrag_plus(
            "python async best practices"
        )
        assert len(results) == 11, (
            "Expected fused results including web/indexed"
        )  # Insight: Web adds 1, graph 1, re-query 9
        assert any("asyncio.gather" in str(r) for r in results), (
            "Expected real-world async insight from web"
        )
        mock_tavily.assert_called_once_with(
            query="python async best practices", max_results=5
        )  # Coverage: Trigger
        mock_qdrant.assert_called()  # Coverage: Re-query after upsert


@pytest.mark.asyncio
async def test_graphrag_plus_sparse_sort() -> None:
    """Test sparse toggle with sorting; real-world: code query for
    'def add(a, b): return a + b' expecting high sparse score."""
    mock_results = [
        {"sparse_score": 0.9, "content": "add function code"},
        {"sparse_score": 0.5, "content": "other"},
    ]
    with (
        patch("..tools.qdrant.query", return_value=mock_results) as mock_qdrant,
        patch("..tools.neo4j_driver.session", new_callable=AsyncMock) as mock_session,
        patch.object(settings, "use_sparse", True),
        patch.object(settings, "use_async", False),
    ):
        mock_session.return_value.run = AsyncMock(
            return_value=[{"n": {"content": "graph add"}}]
        )
        results: list[dict[str, Any]] = await graphrag_plus("add function", "code")
        assert results[0].get("sparse_score") == 0.9, (
            "Expected sorted by sparse score descending"
        )  # Insight: Lexical boost for code
        assert len(results) <= 10, "Expected capped results"
        mock_qdrant.assert_called_once()  # Coverage: Non-async path


@pytest.mark.asyncio
async def test_graphrag_plus_no_web() -> None:
    """Test without web trigger (non-empty vectors); real-world: existing
    'machine learning basics' query."""
    mock_vectors = [{"content": "ML basics: supervised vs unsupervised."}]
    with (
        patch(
            "..tools.qdrant.aquery", new_callable=AsyncMock, return_value=mock_vectors
        ) as mock_qdrant,
        patch("..tools.tavily.search") as mock_tavily,
        patch("..tools.neo4j_driver.session", new_callable=AsyncMock) as mock_session,
        patch.object(settings, "use_async", True),
    ):
        mock_session.return_value.run = AsyncMock(
            return_value=[{"n": {"content": "Graph ML node."}}]
        )
        results: list[dict[str, Any]] = await graphrag_plus("machine learning basics")
        assert len(results) == 2, "Expected vector + graph fuse without web"
        assert "supervised" in str(results[0]), "Expected real-world ML insight"
        mock_tavily.assert_not_called()  # Coverage: No trigger
