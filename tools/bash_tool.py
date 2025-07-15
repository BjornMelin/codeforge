import logging
import subprocess

from crewai.tools import BaseTool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BashTool(BaseTool):
    name: str = "Bash Command Tool"
    description: str = (
        "A secure tool that executes a limited set of bash commands. "
        "It only allows commands starting with 'task-master-ai' to ensure safe execution. "
        "Use this to list tasks, get task details, and update task status."
    )

    def _run(self, command: str) -> str:
        """Executes a bash command if it is on the allowlist ('task-master-ai')."""
        if not command.strip().startswith("task-master-ai"):
            logging.warning(f"Blocked unsafe command: {command}")
            return "Error: Command not allowed. Only 'task-master-ai' commands can be executed."

        try:
            logging.info(f"Executing allowed command: {command}")
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
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


# Instantiate a single instance for agents to use
bash_tool = BashTool()
