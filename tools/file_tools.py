import os

from crewai.tools import tool


@tool("Read File Tool")
def read_file(file_path: str) -> str:
    """Reads the content of a specified file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool("Write File Tool")
def write_file(file_path: str, content: str) -> str:
    """Writes content to a specified file, creating directories if they don't exist."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}."
    except Exception as e:
        return f"Error writing to file: {str(e)}"


@tool("List Directory Tool")
def list_directory(directory_path: str) -> str:
    """Lists the contents of a specified directory."""
    try:
        if not os.path.isdir(directory_path):
            return f"Error: {directory_path} is not a valid directory."
        files = os.listdir(directory_path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing directory: {str(e)}"
