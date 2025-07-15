import yaml

# We will create these agent files in the next step, for now, let's create placeholders
from crewai import Agent, Crew, Task

from agents.coder_agent import coder_agent

# Import all agents and tools
from agents.planner_agent import planner_agent
from tools.bash_tool import bash_tool
from tools.file_tools import read_file, write_file
from tools.git_tools import git_diff
from tools.shipit_tools import (
    create_incremental_commit,
    create_pull_request,
    quality_check,
)

researcher_agent = Agent(
    role="placeholder", goal="placeholder", backstory="placeholder"
)
tester_agent = Agent(role="placeholder", goal="placeholder", backstory="placeholder")
deployer_agent = Agent(role="placeholder", goal="placeholder", backstory="placeholder")


class YAMLWorkflowEngine:
    def __init__(self, workflow_file: str):
        with open(workflow_file, "r") as f:
            self.workflow_def = yaml.safe_load(f)

        self._load_agents_and_tools()

    def _load_agents_and_tools(self):
        """Create mappings from names to actual agent and tool objects."""
        self.agents = {
            "Planner": planner_agent,
            "Coder": coder_agent,
            "Researcher": researcher_agent,
            "Tester": tester_agent,
            "Deployer": deployer_agent,
        }
        self.tools = {
            "bash_tool": bash_tool,
            "read_file": read_file,
            "write_file": write_file,
            "git_diff": git_diff,
            "quality_check": quality_check,
            "create_incremental_commit": create_incremental_commit,
            "create_pull_request": create_pull_request,
        }

    def create_crew(self, task_params: dict) -> Crew:
        """Dynamically creates the crew and tasks from the YAML definition."""
        tasks = []
        for step in self.workflow_def.get("steps", []):
            agent = self.agents.get(step["agent"])
            if not agent:
                raise ValueError(f"Agent '{step['agent']}' not found in agent mapping.")

            # Substitute parameters into the description
            description = step["description"].format(**task_params)

            task = Task(
                description=description,
                expected_output=step["expected_output"],
                agent=agent,
                tools=[self.tools[t] for t in step.get("tools", [])],
            )
            tasks.append(task)

        return Crew(agents=list(self.agents.values()), tasks=tasks, verbose=True)

    def kickoff(self, task_params: dict):
        """Creates and kicks off the crew execution."""
        crew = self.create_crew(task_params)
        return crew.kickoff()
