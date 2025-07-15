from crewai import Agent
from langchain_openai import ChatOpenAI

from tools.file_tools import read_file, write_file
from tools.git_tools import git_diff
from tools.shipit_tools import create_incremental_commit, quality_check

# Define the LLM for the Coder Agent - a powerful model for code generation.
coder_llm = ChatOpenAI(model_name="moonshotai/kimi-k2", temperature=0.15)

# Create the Coder Agent
coder_agent = Agent(
    role="Principal Software Engineer",
    goal=(
        "Write clean, efficient, and high-quality code to implement the features "
        "outlined in the implementation plan. Your primary workflow is to: "
        "1. Implement a small, logical piece of the plan (e.g., one function, one file). "
        "2. Use the 'Code Quality and Formatting Tool' on the files you have changed. "
        "3. **CRITICAL**: If the quality check fails, you MUST analyze the errors, fix the code, and re-run the tool. "
        "   This is the Bounded Self-Correction loop. You have a maximum of 3 attempts to fix the errors for a given change. "
        "   If you fail after 3 attempts, stop and report the final, unfixable error. "
        "4. Once the quality check passes (returns 'Success'), use the 'Incremental Git Commit Tool' to commit your changes. "
        "   The commit message MUST follow the Conventional Commits standard. "
        "   The 'scope' of the commit should be dynamically inferred from the primary directory of your changes (e.g., 'auth', 'ui', 'api'). "
        "5. Repeat this loop until the entire implementation plan is complete."
    ),
    backstory=(
        "You are a seasoned software engineer with a passion for writing robust and maintainable code. "
        "You believe in a strict, quality-first development process. You write code in small, "
        "logical increments and ensure each increment is validated and committed before moving on. "
        "You are skilled at debugging and fixing your own code based on linter feedback."
    ),
    tools=[read_file, write_file, git_diff, quality_check, create_incremental_commit],
    llm=coder_llm,
    allow_delegation=False,
    verbose=True,
)
