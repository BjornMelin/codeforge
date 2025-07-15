from crewai import Agent
from langchain_openai import ChatOpenAI
from tools.bash_tool import bash_tool

# Define the LLM for the Planner Agent - a fast and cost-effective model.
# Assumes OPENAI_API_KEY and OPENAI_API_BASE (for OpenRouter) are in .env
# TODO: Use direct OpenAI API for OpenAI Models
planner_llm = ChatOpenAI(model_name="openai/o3", temperature=0.0)

# Create the Planner Agent
planner_agent = Agent(
    role="Lead Software Planner",
    goal=(
        "Analyze requirements, find the next priority task from the task management system, "
        "and create a detailed, step-by-step implementation plan for the engineering team."
    ),
    backstory=(
        "With a keen eye for detail and a strategic mindset, you excel at breaking down "
        "complex software requirements into actionable, sequential tasks. You are the "
        "first point of contact in the development workflow, ensuring the team has a "
        "clear and unambiguous plan before any code is written."
    ),
    tools=[bash_tool],
    llm=planner_llm,
    allow_delegation=False,
    verbose=True
)