"""
Data Copilot Lab - FastAPI Main Application
Entry point for the REST API
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.exceptions import DataCopilotException
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    Handles startup and shutdown events
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    # Ensure directories exist
    settings.ensure_directories()
    logger.info("Application directories verified")

    yield

    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered Data Science Platform for end-to-end analytics workflow",
    lifespan=lifespan,
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# Exception Handlers
@app.exception_handler(DataCopilotException)
async def data_copilot_exception_handler(
    request: Request,
    exc: DataCopilotException
):
    """Handle custom Data Copilot exceptions"""
    logger.error(
        f"DataCopilotException: {exc.message}",
        extra={"code": exc.code, "details": exc.details}
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.code,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An internal server error occurred"
        }
    )


# Health Check Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "environment": settings.environment
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment
    }


@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "features": {
            "ai_assistant": settings.enable_ai_assistant,
            "automl": settings.enable_automl,
            "deep_learning": settings.enable_deep_learning,
            "profiling": settings.enable_profiling,
        }
    }


# Include routers
from src.api.routes import data

app.include_router(data.router, prefix="/api/data", tags=["Data Import"])

# Future routers (will be added as we implement modules)
# from src.api.routes import analysis, ml, ai, reports
# app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
# app.include_router(ml.router, prefix="/api/ml", tags=["Machine Learning"])
# app.include_router(ai.router, prefix="/api/ai", tags=["AI Assistant"])
# app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
