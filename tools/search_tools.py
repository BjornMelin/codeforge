import json
import os

import httpx
from crewai.tools import tool

# It's good practice to manage API keys centrally, e.g., via environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


@tool("Web Search Tool")
def search_tool(query: str) -> str:
    """
    Performs a web search using the Tavily API to find information on a given topic.
    This is the primary tool for all research tasks.
    """
    if not TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY environment variable is not set."

    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": 5,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        results = response.json()
        # Return a formatted string of the results for the agent to process
        return json.dumps(results.get("results", []), indent=2)
    except httpx.HTTPStatusError as e:
        return f"Error performing search: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"An unexpected error occurred during search: {str(e)}"
