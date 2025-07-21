# coding=utf-8
"""Model routing for CodeForge AI.

This module handles dynamic model selection and invocation via OpenRouter.
"""

from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from .config import settings

openrouter: AsyncOpenAI = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key
)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def route_model(task: str, category: str) -> dict[str, Any]:
    """Route task to appropriate model based on complexity and category.

    Args:
        task: Task description string.
        category: Task category (e.g., "reasoning", "coding").

    Returns:
        Dictionary with selected model and response.
    """
    if "complex" in task or category == "reasoning":
        model: str = "xai/grok-4"
    elif "coding" in task:
        model = "anthropic/claude-4-sonnet"
    elif "research" in task:
        model = "kimi/k2"
    else:
        model = "google/gemini-2.5-flash"

    response_format: dict[str, Any] | None = (
        {
            "type": "json_object",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                },
            },
        }
        if settings.use_structured
        else None
    )

    response = await openrouter.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": task}],
        max_tokens=500,
        response_format=response_format,
    )
    return {"model": model, "response": response.choices[0].message.content}
