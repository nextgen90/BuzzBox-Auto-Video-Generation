import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from loguru import logger

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router as api_router
from app.utils.helpers import sanitize_for_json
from app.services.storage_service import StorageService
from app.core.exceptions import BuzzBoxException

# Configure rotating logger file and stdout handlers
setup_logging()

from fastapi.staticfiles import StaticFiles
from app.engines.video.model_manager import ModelManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle event manager to handle server startup initialization
    and clean shutdown sequences.
    """
    logger.info("Starting up BuzzBox AI Engine Backend...")
    # Initialize workspace directories using StorageService
    try:
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # This will create characters, jobs, prompts, cache, exports, renders, etc.
        storage_service = StorageService()
        logger.info("Successfully verified and initialized all backend storage directories.")
    except Exception as e:
        logger.error(f"Error initializing workspace directories during startup: {str(e)}")

    # Load and warm up the selected model engine
    try:
        model_manager = ModelManager()
        logger.info("Pre-loading active video generation engine...")
        model_manager.load()
        logger.info("Running engine warmup pass...")
        model_manager.warmup()
        logger.info("AI Model Engine loaded, warmed up, and ready.")
    except Exception as e:
        logger.error(f"Failed to load/warmup AI Model Engine during startup: {str(e)}")
        
    yield
    
    logger.info("Shutting down BuzzBox AI Engine Backend...")

# Instantiate FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise AI Video Generation Backend",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Mount Local S3 Mock static files route
s3_mock_dir = settings.STORAGE_DIR / "s3_mock"
s3_mock_dir.mkdir(parents=True, exist_ok=True)
app.mount("/s3_mock", StaticFiles(directory=str(s3_mock_dir)), name="s3_mock")

# Apply CORS Middleware to support frontend and integration architectures
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging and execution timer middleware
@app.middleware("http")
async def request_logger_and_timer(request: Request, call_next):
    start_time = time.perf_counter()
    method = request.method
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"Ingressing request: {method} {path} from IP: {client_ip}")

    try:
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{duration:.4f}s"
        
        logger.info(
            f"Egressing response: {method} {path} | Status: {response.status_code} | Duration: {duration:.4f}s"
        )
        return response
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.exception(
            f"Unhandled exception during: {method} {path} | Duration: {duration:.4f}s | Detail: {str(e)}"
        )
        raise

# Custom Exception Handler for custom BuzzBox exceptions
@app.exception_handler(BuzzBoxException)
async def buzzbox_exception_handler(request: Request, exc: BuzzBoxException):
    logger.warning(f"BuzzBox Exception {exc.error_code} on {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": exc.error_code,
            "message": exc.message
        }
    )

# Custom Global Exception Handler for Starlette/FastAPI HTTP Exceptions (e.g. 404 Not Found)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(f"HTTP Error {exc.status_code} on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail
        }
    )

# Custom Global Exception Handler for Request Validation Errors (422 Unprocessable Entity)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    logger.warning(f"Validation Error 422 on {request.url.path}: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "The request payload failed schema validation.",
            "details": sanitize_for_json(errors)
        }
    )

# Custom Global Exception Handler for general unhandled exceptions (500 Internal Server Error)
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Critical 500 Unhandled Exception on {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred while processing the request."
        }
    )

# Register the standard routing configurations
app.include_router(api_router)
