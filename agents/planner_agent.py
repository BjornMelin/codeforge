from crewai import Agent, Task, Crew

from tools.bash_tool import bash_tool

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
    allow_delegation=False,
    verbose=True,
)

# To allow for direct testing of the agent
if __name__ == "__main__":
    print("Testing Planner Agent...")

    # Define a test task for the agent
    planning_task = Task(
        description=(
            "Find the next pending task from the task management system. "
            "Use the `task-master-ai list --status pending --json` command. "
            "Once you have the task list, identify the highest priority task and "
            "output its ID and title."
        ),
        expected_output="The ID and title of the highest priority pending task.",
        agent=planner_agent,
    )

    # To make this test runnable, we need a dummy task-master-ai script
    with open("task-master-ai", "w") as f:
        f.write("""
#!/bin/bash
echo '{
  "tasks": [
    {"id": 101, "title": "Implement OAuth2 login", "priority": "high", "status": "pending"},
    {"id": 102, "title": "Fix caching bug", "priority": "medium", "status": "pending"}
  ]
}'
        """)
    import os

    os.chmod("task-master-ai", 0o755)

    # Add the current directory to the PATH for the agent's subprocess to find the script
    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f".:{original_path}"

    # Execute the task using a crew
    test_crew = Crew(
        agents=[planner_agent],
        tasks=[planning_task],
        verbose=True
    )
    
    result = test_crew.kickoff()

    # Restore original path
    os.environ["PATH"] = original_path

    print("\n--- Planner Agent Test Result ---")
    print(result)

    # Cleanup dummy script
    os.remove("task-master-ai")
