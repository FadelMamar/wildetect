"""FastAPI application entry point for WildTrain API."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
from typing import Dict, Any
import typer
import traceback

from .routers import training, evaluation, pipeline, visualization, dataset, config
from .utils.error_handling import WildTrainAPIException, wildtrain_exception_handler
from .dependencies import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="WildTrain API",
        description="REST API for WildTrain - Modular Computer Vision Framework",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom exception handler
    app.add_exception_handler(WildTrainAPIException, wildtrain_exception_handler)
    
    # Include routers
    app.include_router(training.router, prefix="/training", tags=["Training"])
    app.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation"])
    app.include_router(pipeline.router, prefix="/pipeline", tags=["Pipeline"])
    app.include_router(visualization.router, prefix="/visualization", tags=["Visualization"])
    app.include_router(dataset.router, prefix="/dataset", tags=["Dataset"])
    app.include_router(config.router, prefix="/config", tags=["Configuration"])
    
    @app.get("/")
    async def root() -> Dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "message": "WildTrain API",
            "version": "0.1.0",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler for unexpected errors."""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "error": str(exc) if settings.debug else "An unexpected error occurred"
            }
        )
    
    try:
        from fastapi_mcp import FastApiMCP

        mcp = FastApiMCP(
        app,
        name="WildTrain API MCP",
        description="WildTrain API MCP server",
        describe_all_responses=True,
        describe_full_response_schema=True,
        )
        mcp.mount_http()
        mcp.setup_server()
    except ImportError:
        logger.warning("fastapi_mcp is not installed, MCP server will not be available")
    except Exception:
        logger.error(f"Error initializing MCP server: {traceback.format_exc()}")

    return app

# Create the app instance
fastapi_app = create_app()

cli_app = typer.Typer(name="api", help="WildTrain API server commands")

@cli_app.command()
def serve(
    host: str = typer.Option(settings.host, "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(settings.port, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(
        settings.debug, "--reload", "-r", help="Enable auto-reload"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Number of worker processes"
    ),
):
    """Start the Wildtrain API server."""
    import uvicorn

    typer.echo(f"Starting Wildtrain API server on {host}:{port}")
    typer.echo(f"Debug mode: {reload}")
    typer.echo(f"Workers: {workers}")
    typer.echo(f"API documentation: http://{host}:{port}/docs")

    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )


@cli_app.command()
def check():
    """Check API configuration."""
    typer.echo("WildTraina API Configuration:")
    typer.echo(f"  Host: {settings.host}")
    typer.echo(f"  Port: {settings.port}")
    typer.echo(f"  Debug: {settings.debug}")
    typer.echo(f"  Upload directory: {settings.upload_dir}")
    typer.echo(f"  CORS origins: {settings.cors_origins}")