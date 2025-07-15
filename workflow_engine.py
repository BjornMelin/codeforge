import yaml
from crewai import Crew, Task

from agents.coder_agent import coder_agent
from agents.deployer_agent import deployer_agent

# CHANGE: Import the actual agent definitions, replacing the placeholders.
from agents.planner_agent import planner_agent
from agents.researcher_agent import researcher_agent
from agents.tester_agent import tester_agent
from tools.bash_tool import bash_tool
from tools.file_tools import read_file, write_file
from tools.git_tools import git_diff
from tools.quality_tools import run_comprehensive_tests
from tools.search_tools import search_tool
from tools.shipit_tools import (
    create_incremental_commit,
    create_pull_request,
    quality_check,
)


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
            "search_tool": search_tool,
            "run_comprehensive_tests": run_comprehensive_tests,
        }

    def create_crew(self, task_params: dict) -> Crew:
        """Dynamically creates the crew and tasks from the YAML definition."""
        tasks = []
        for step in self.workflow_def.get("steps", []):
            agent = self.agents.get(step["agent"])
            if not agent:
                raise ValueError(f"Agent '{step['agent']}' not found in agent mapping.")

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
