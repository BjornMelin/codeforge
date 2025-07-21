# coding=utf-8
"""Tools for retrieval and RAG in CodeForge AI.

This module implements advanced GraphRAG+ with hybrid DB and web integration.
"""

from typing import Any, Optional

from neo4j import AsyncGraphDatabase, GraphDatabase
from qdrant_client import AsyncQdrantClient, QdrantClient
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_fixed

from .config import settings

qdrant: Optional[AsyncQdrantClient | QdrantClient] = (
    AsyncQdrantClient(url=settings.qdrant_url)
    if settings.use_async
    else QdrantClient(url=settings.qdrant_url)
)
neo4j_driver: Optional[AsyncGraphDatabase | GraphDatabase] = (
    AsyncGraphDatabase.driver(settings.neo4j_uri, auth=("neo4j", "password"))
    if settings.use_async
    else GraphDatabase.driver(settings.neo4j_uri, auth=("neo4j", "password"))
)
tavily: TavilyClient = TavilyClient(api_key=settings.tavily_api_key)
embedder: SentenceTransformer = SentenceTransformer("BAAI/bge-m3", device="cpu")
sparse_embedder: Optional[SentenceTransformer] = (
    SentenceTransformer("BAAI/bge-sparse-en-v1.5") if settings.use_sparse else None
)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def graphrag_plus(
    query: str, content_type: str = "general"
) -> list[dict[str, Any]]:
    """Perform agentic hybrid GraphRAG+ retrieval with web fallback.

    Args:
        query: Search query string.
        content_type: Type of content for embedding variation (default: "general").

    Returns:
        List of fused retrieval results.
    """
    dim: int = 256 if content_type == "code" else 768
    query_embed: list[float] = embedder.encode(query)[:dim].tolist()
    sparse_query: Optional[list[float]] = (
        sparse_embedder.encode(query).tolist() if sparse_embedder else None
    )

    if settings.use_async and isinstance(qdrant, AsyncQdrantClient):
        vector_results = await qdrant.aquery(
            collection_name="docs",
            query=query_embed,
            limit=5,
            sparse_vector=sparse_query,
        )
    else:
        vector_results = qdrant.query(  # type: ignore
            collection_name="docs",
            query=query_embed,
            limit=5,
            sparse_vector=sparse_query,
        )

    if not vector_results:
        web_results: list[dict[str, Any]] = tavily.search(query=query, max_results=5)
        content: str = web_results[0]["content"]
        web_embed: list[float] = embedder.encode(content).tolist()
        web_sparse: Optional[list[float]] = (
            sparse_embedder.encode(content).tolist() if sparse_embedder else None
        )

        if settings.use_async and isinstance(neo4j_driver, AsyncGraphDatabase):
            async with neo4j_driver.session() as session:
                await session.run(
                    "CREATE (n:WebResult {content: $content})", content=content
                )
            await qdrant.aupsert(
                collection_name="docs",
                points=[
                    {"id": "web1", "vector": web_embed, "sparse_vector": web_sparse}
                ],
            )
            vector_results = await qdrant.aquery(
                collection_name="docs",
                query=query_embed,
                limit=5,
                sparse_vector=sparse_query,
            )
        else:
            with neo4j_driver.session() as session:  # type: ignore
                session.run("CREATE (n:WebResult {content: $content})", content=content)
            qdrant.upsert(  # type: ignore
                collection_name="docs",
                points=[
                    {"id": "web1", "vector": web_embed, "sparse_vector": web_sparse}
                ],
            )
            vector_results = qdrant.query(  # type: ignore
                collection_name="docs",
                query=query_embed,
                limit=5,
                sparse_vector=sparse_query,
            )

    if settings.use_async and isinstance(neo4j_driver, AsyncGraphDatabase):
        async with neo4j_driver.session() as session:
            graph_results: list[dict[str, Any]] = (
                await session.run(
                    "MATCH (n) WHERE n.content CONTAINS $query RETURN n", query=query
                )
            ).data()
    else:
        with neo4j_driver.session() as session:  # type: ignore
            graph_results: list[dict[str, Any]] = session.run(
                "MATCH (n) WHERE n.content CONTAINS $query RETURN n", query=query
            ).data()

    fused: list[dict[str, Any]] = vector_results + graph_results  # type: ignore
    if settings.use_sparse:
        fused = sorted(fused, key=lambda x: x.get("sparse_score", 0), reverse=True)
    return fused[:10]
