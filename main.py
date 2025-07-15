import click
from workflow_engine import YAMLWorkflowEngine


@click.group()
def cli():
    """CodeForge AI: Autonomous Multi-Agent Coding System"""
    pass


@cli.command()
@click.option(
    "--feature",
    "feature_description",
    required=True,
    help="A description of the feature to be implemented.",
)
def ship(feature_description: str):
    """Triggers the end-to-end autonomous feature implementation workflow."""
    click.echo(f"🚀 Initiating autonomous workflow to implement: {feature_description}")

    workflow_file = "workflows/feature_implementation.yaml"
    engine = YAMLWorkflowEngine(workflow_file)

    task_params = {"feature_description": feature_description}

    result = engine.kickoff(task_params)

    click.echo("\n--- Workflow Execution Summary ---")
    click.echo(result)


@cli.command()
@click.option(
    "--name",
    "workflow_name",
    required=True,
    help="The name of the YAML workflow to run (e.g., bug_investigation).",
)
@click.option(
    "--params",
    "params_json",
    required=False,
    default="{}",
    help="A JSON string of parameters for the workflow.",
)
def run_workflow(workflow_name: str, params_json: str):
    """Runs a specific, pre-defined YAML workflow."""
    import json

    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        click.echo("Error: Invalid JSON provided for --params.", err=True)
        return

    click.echo(f"🚀 Running workflow: {workflow_name}")

    workflow_file = f"workflows/{workflow_name}.yaml"
    engine = YAMLWorkflowEngine(workflow_file)

    result = engine.kickoff(params)

    click.echo("\n--- Workflow Execution Summary ---")
    click.echo(result)


if __name__ == "__main__":
    cli()
