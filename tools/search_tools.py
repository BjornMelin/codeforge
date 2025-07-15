import json
import os

from crewai.tools import tool
from tavily import TavilyClient

# Ensure TAVILY_API_KEY is set in your .env file
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Instantiate the client once
if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
else:
    tavily_client = None


@tool("Web Search Tool")
def search_tool(query: str) -> str:
    """
    Performs a web search using the Tavily API to find information on a given topic.
    This is the primary tool for all research tasks.
    """
    if not tavily_client:
        return "Error: TAVILY_API_KEY environment variable is not set."

    try:
        # Use the official Tavily client library
        results = tavily_client.search(
            query=query, search_depth="advanced", max_results=5
        )
        # The library returns a dictionary; we'll dump it to a JSON string for the agent
        return json.dumps(results.get("results", []), indent=2)
    except Exception as e:
        # The library will handle specific HTTP errors, but we catch any other exceptions
        return f"An unexpected error occurred during search: {str(e)}"
