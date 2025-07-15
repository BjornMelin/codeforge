import logging
import subprocess

from crewai.tools import tool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@tool("Comprehensive Test Execution Tool")
def run_comprehensive_tests(directory: str = "./tests") -> str:
    """
    Runs the full suite of automated tests (e.g., pytest) in a specified directory.
    """
    logging.info(f"Executing comprehensive tests in directory: {directory}")
    try:
        # Assuming pytest is the testing framework. This can be adapted.
        command = ["pytest", directory]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,  # 5-minute timeout for tests
        )
        logging.info("Comprehensive tests passed.")
        return f"Success: All tests passed.\n{result.stdout}"
    except FileNotFoundError:
        return "Error: 'pytest' command not found. Please ensure pytest is installed."
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Tests failed with exit code {e.returncode}:\n{e.stdout}\n{e.stderr}"
        )
        logging.error(error_message)
        return error_message
    except subprocess.TimeoutExpired:
        return "Error: Test execution timed out after 5 minutes."
    except Exception as e:
        return f"An unexpected error occurred during testing: {str(e)}"
