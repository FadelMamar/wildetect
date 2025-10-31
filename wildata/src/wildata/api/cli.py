"""
CLI commands for the API server.
"""

from pathlib import Path

import typer

from .config import api_config

app = typer.Typer(name="api", help="WildData API server commands")


@app.command()
def serve(
    host: str = typer.Option(api_config.host, "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(api_config.port, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(
        api_config.debug, "--reload", "-r", help="Enable auto-reload"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Number of worker processes"
    ),
):
    """Start the WildData API server."""
    import uvicorn

    typer.echo(f"Starting WildData API server on {host}:{port}")
    typer.echo(f"Debug mode: {reload}")
    typer.echo(f"Workers: {workers}")
    typer.echo(f"API documentation: http://{host}:{port}/docs")

    uvicorn.run(
        "wildata.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )


@app.command()
def check():
    """Check API configuration."""
    typer.echo("WildData API Configuration:")
    typer.echo(f"  Host: {api_config.host}")
    typer.echo(f"  Port: {api_config.port}")
    typer.echo(f"  Debug: {api_config.debug}")
    typer.echo(f"  Upload directory: {api_config.upload_dir}")
    typer.echo(f"  Max file size: {api_config.max_file_size} bytes")
    typer.echo(f"  Job queue size: {api_config.job_queue_size}")
    typer.echo(f"  CORS origins: {api_config.cors_origins}")
