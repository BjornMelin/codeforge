import logging
import re
import subprocess

from crewai.tools import tool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Logic adapted from quality-gate-validator.py ---
SECRET_PATTERNS = [
    r'(?i)(?:api[_-]?key|apikey|secret|password|token)\s*[=:]\s*["\'][\w\-]{16,}["\']',
    r"(?i)bearer\s+[\w\-\.]+",
]


def _is_placeholder(value: str) -> bool:
    """Checks if a matched secret is a placeholder."""
    value_match = re.search(r'["\']([^"\']+)["\']', value)
    actual_value = value_match.group(1).lower() if value_match else value.lower()
    placeholders = [
        "your-api-key",
        "placeholder",
        "example",
        "test-key",
        "dummy",
        "xxx",
    ]
    return any(p in actual_value for p in placeholders)


def _scan_for_secrets(content: str) -> list[str]:
    """Scans content for hardcoded secrets."""
    found_secrets = []
    for pattern in SECRET_PATTERNS:
        matches = re.findall(pattern, content)
        for match in matches:
            if not _is_placeholder(match):
                found_secrets.append(f"Potential secret detected: {match[:30]}...")
    return found_secrets


# --- End of adapted logic ---


@tool("Code Quality and Formatting Tool")
def quality_check(file_paths: str) -> str:
    """
    Runs a suite of quality checks (linting, formatting, secret scanning) on a list of files.
    The input must be a string of file paths separated by spaces.
    Returns 'Success' if all checks pass, otherwise returns a formatted string of errors.
    """
    files = file_paths.split()
    logging.info(f"Running quality checks on: {files}")
    all_errors = []

    # 1. Formatting and Linting with Ruff
    try:
        format_command = ["ruff", "format"] + files
        subprocess.run(format_command, check=True, capture_output=True, text=True)

        lint_command = ["ruff", "check", "--fix", "--exit-zero"] + files
        lint_result = subprocess.run(
            lint_command, check=True, capture_output=True, text=True
        )
        if lint_result.stdout:
            all_errors.append(f"Ruff linting issues found:\n{lint_result.stdout}")
    except FileNotFoundError:
        all_errors.append(
            "Error: 'ruff' command not found. Please ensure it is installed."
        )
    except subprocess.CalledProcessError as e:
        all_errors.append(f"Ruff command failed:\n{e.stderr}")
    except Exception as e:
        all_errors.append(
            f"An unexpected error occurred during linting/formatting: {str(e)}"
        )

    # 2. Secret Scanning
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            secrets = _scan_for_secrets(content)
            if secrets:
                all_errors.extend(secrets)
        except FileNotFoundError:
            all_errors.append(f"Error: File not found for secret scan: {file_path}")
        except Exception as e:
            all_errors.append(f"Error scanning {file_path} for secrets: {str(e)}")

    if not all_errors:
        logging.info("All quality checks passed.")
        return "Success: All quality checks passed."
    else:
        error_report = "Quality checks failed:\n- " + "\n- ".join(all_errors)
        logging.warning(error_report)
        return error_report


@tool("Incremental Git Commit Tool")
def create_incremental_commit(message: str) -> str:
    """
    Stages all changes and creates a git commit with the provided message.
    """
    logging.info(f"Creating incremental commit with message: '{message}'")
    try:
        subprocess.run(["git", "add", "."], check=True)
        commit_result = subprocess.run(
            ["git", "commit", "-m", message], check=True, capture_output=True, text=True
        )
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        logging.info(f"Successfully created commit {commit_hash}")
        return f"Success: Commit '{commit_hash}' created.\n{commit_result.stdout}"
    except subprocess.CalledProcessError as e:
        error_message = f"Git commit failed: {e.stderr}"
        if "nothing to commit" in error_message:
            return "Warning: No changes to commit."
        logging.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"An unexpected error occurred during commit: {str(e)}"
        logging.error(error_message)
        return error_message


@tool("GitHub Pull Request Tool")
def create_pull_request(title: str, body: str, branch: str) -> str:
    """
    Pushes the current branch to the remote and creates a pull request on GitHub.
    """
    logging.info(f"Creating pull request titled: '{title}'")
    try:
        _push_result = subprocess.run(
            ["git", "push", "-u", "origin", branch],
            check=True,
            capture_output=True,
            text=True,
        )
        logging.info(f"Branch '{branch}' pushed to remote.")

        pr_command = [
            "gh",
            "pr",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--head",
            branch,
            "--base",
            "main",
        ]
        pr_result = subprocess.run(
            pr_command, check=True, capture_output=True, text=True
        )
        pr_url = pr_result.stdout.strip()
        logging.info(f"Successfully created pull request: {pr_url}")
        return f"Success: Pull request created at {pr_url}"
    except FileNotFoundError:
        return "Error: 'gh' command not found. Please ensure the GitHub CLI is installed and authenticated."
    except subprocess.CalledProcessError as e:
        error_message = f"Pull request creation failed: {e.stderr}"
        logging.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"An unexpected error occurred during PR creation: {str(e)}"
        logging.error(error_message)
        return error_message
