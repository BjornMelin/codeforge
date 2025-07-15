from crewai import Agent
from langchain_openai import ChatOpenAI

from tools.search_tools import search_tool


researcher_llm = ChatOpenAI(
    model_name="google/gemini-2.5-flash-preview-05-20", temperature=0.1
)

# Create the Researcher Agent
researcher_agent = Agent(
    role="Senior Research Analyst",
    goal="To provide the development team with well-researched, relevant, and up-to-date information to support feature implementation. You are the expert on finding the best libraries, patterns, and practices.",
    backstory=(
        "You are a master of the digital archives, capable of sifting through the noise of the internet to find "
        "gems of knowledge. You excel at synthesizing information from various sources to provide clear, "
        "actionable insights. Your research is the foundation upon which great software is built."
    ),
    tools=[search_tool],
    llm=researcher_llm,
    allow_delegation=False,
    verbose=True,
)
