import subprocess

from crewai.tools import tool


@tool("Git Diff Tool")
def git_diff(file_path: str = None) -> str:
    """
    Shows the git diff for the repository. Can be for a specific file or all staged changes.
    """
    try:
        command = [
            "git",
            "diff",
            "--staged",
        ]  # Focus on staged changes for commit context
        if file_path:
            command.append(file_path)

        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout if result.stdout else "No staged changes."
    except subprocess.CalledProcessError as e:
        return f"Error getting git diff: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
