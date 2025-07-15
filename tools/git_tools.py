import subprocess

from crewai.tools import tool


@tool("Git Diff Tool")
def git_diff(file_path: str = None) -> str:
    """
    Shows the git diff for staged changes. Can be for a specific file or all changes.
    This is crucial for reviewing work before committing.
    """
    try:
        command = ["git", "diff", "--staged"]
        if file_path:
            command.append(file_path)

        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout if result.stdout else "No staged changes to show."
    except subprocess.CalledProcessError as e:
        return f"Error getting git diff: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


@tool("List Repository Files Tool")
def list_repository_files() -> str:
    """Lists all files in the git repository, respecting .gitignore."""
    try:
        result = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error listing repository files: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
