import logging
import subprocess

from crewai.tools import tool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# CHANGE: Refactored from a class inheriting from BaseTool to a standalone function
# decorated with @tool. This is the modern, simpler way to create tools in CrewAI.
@tool("Bash Command Tool")
def bash_tool(command: str) -> str:
    """
    A secure tool that executes a limited set of bash commands.
    It only allows commands starting with 'task-master-ai' to ensure safe execution.
    Use this to list tasks, get task details, and update task status.

    Args:
        command (str): The bash command to execute.

    Returns:
        str: The stdout of the command if successful, or an error message.
    """
    # Security Gate: Only allow commands starting with 'task-master-ai'
    if not command.strip().startswith("task-master-ai"):
        logging.warning(f"Blocked unsafe command: {command}")
        return "Error: Command not allowed. Only 'task-master-ai' commands can be executed."

    try:
        logging.info(f"Executing allowed command: {command}")
        # Using shell=True is safe here due to the strict allowlist check above.
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,  # 60-second timeout for safety
        )
        logging.info(f"Command successful. STDOUT: {result.stdout[:100]}...")
        return result.stdout
    except subprocess.TimeoutExpired:
        logging.error(f"Command timed out: {command}")
        return f"Error: Command '{command}' timed out after 60 seconds."
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Command failed with exit code {e.returncode}: {command}\nSTDERR: {e.stderr}"
        )
        return f"Error executing command: {e.stderr}"
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {str(e)}"


# To allow for direct testing of the tool
if __name__ == "__main__":
    print("Testing BashTool...")

    # Test Case 1: Allowed command
    print("\n--- Testing allowed command ---")
    # Create a dummy task-master-ai script for testing purposes
    with open("task-master-ai", "w") as f:
        f.write('#!/bin/bash\necho \'{"tasks": [{"id": 1, "title": "Test Task"}]}\'')
    import os

    os.chmod("task-master-ai", 0o755)

    # The path needs to be adjusted if the script is not in the same directory
    # For this test, we assume it's run from the project root.
    allowed_command = "./task-master-ai list --status pending --json"
    output = bash_tool.run(command=allowed_command)
    print(f"Command: '{allowed_command}'\nOutput: {output}")

    # Test Case 2: Disallowed command
    print("\n--- Testing disallowed command ---")
    disallowed_command = "ls -la"
    output = bash_tool.run(command=disallowed_command)
    print(f"Command: '{disallowed_command}'\nOutput: {output}")

    # Cleanup dummy script
    os.remove("task-master-ai")
