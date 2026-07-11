from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from loguru import logger
from app.models.request import VideoGenerationRequest
from app.models.response import VideoGenerationResponse
from app.services.video_service import VideoService
from app.services.job_service import JobService
from app.services.character_service import CharacterService
from app.services.runtime_service import RuntimeService
from app.workers.generation_worker import process_generation_job
from app.core.config import settings
from app.core.exceptions import BuzzBoxException
from app.engines.video.model_manager import ModelManager
from app.core.gpu_scheduler import GPUScheduler

router = APIRouter()

# Dependency Injection Helpers
def get_video_service() -> VideoService:
    return VideoService()

def get_job_service() -> JobService:
    return JobService()

def get_character_service() -> CharacterService:
    return CharacterService()

def get_runtime_service() -> RuntimeService:
    return RuntimeService()


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Root Endpoint",
    description="Returns application branding information and operational state.",
    tags=["General"]
)
async def read_root():
    return {
        "application": settings.APP_NAME,
        "status": "running"
    }


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check Endpoint",
    description="Assess system diagnostics status including hardware and storage verification.",
    tags=["General"]
)
async def read_health(
    runtime_service: RuntimeService = Depends(get_runtime_service),
    character_service: CharacterService = Depends(get_character_service),
    job_service: JobService = Depends(get_job_service)
):
    cuda_avail = runtime_service.is_cuda_available()
    torch_ver = runtime_service.torch_version()
    
    # Check directory access
    char_store_status = "ready"
    try:
        settings.CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        char_store_status = "unwritable"

    jobs_store_status = "ready"
    try:
        settings.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        jobs_store_status = "unwritable"

    return {
        "application": settings.APP_NAME,
        "status": "healthy" if (char_store_status == "ready" and jobs_store_status == "ready") else "degraded",
        "gpu": "ready" if cuda_avail else "cpu_fallback",
        "cuda": cuda_avail,
        "torch": torch_ver,
        "character_storage": char_store_status,
        "jobs": jobs_store_status,
        "version": settings.APP_VERSION
    }


@router.post(
    "/generate-video",
    response_model=VideoGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest Video Generation Request",
    description="Validates parameters, registers a job, dispatches it to background worker, and returns immediately.",
    tags=["Video Generation"]
)
async def generate_video(
    payload: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
    video_service: VideoService = Depends(get_video_service)
):
    logger.info(f"Received request to queue video generation. Client-Request-ID: {payload.request_id or 'none'}")
    try:
        response = video_service.create_job(payload)
        # Dispatch to background task runner
        background_tasks.add_task(process_generation_job, response.job_id)
        logger.info(f"Request successfully queued & worker dispatched. Job-ID: {response.job_id}")
        return response
    except ValueError as ve:
        logger.warning(f"Domain validation failed: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except BuzzBoxException as bbe:
        raise HTTPException(
            status_code=bbe.status_code,
            detail={"error_code": bbe.error_code, "message": bbe.message}
        )
    except Exception as e:
        logger.error(f"Failed to create video generation job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while queueing the video generation task."
        )


@router.get(
    "/jobs/{job_id}",
    status_code=status.HTTP_200_OK,
    summary="Get Job Status",
    description="Retrieves granular real-time progress, estimated remaining time, and logs for a specific job.",
    tags=["Job Engine"]
)
async def get_job(
    job_id: str,
    job_service: JobService = Depends(get_job_service)
):
    try:
        return job_service.get_job_status(job_id)
    except BuzzBoxException as bbe:
        raise HTTPException(
            status_code=bbe.status_code,
            detail={"error_code": bbe.error_code, "message": bbe.message}
        )
    except Exception as e:
        logger.error(f"Error fetching job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching job: {str(e)}"
        )


@router.get(
    "/jobs",
    status_code=status.HTTP_200_OK,
    summary="List Jobs",
    description="Returns all registered jobs in storage sorted newest first.",
    tags=["Job Engine"]
)
async def list_jobs(
    job_service: JobService = Depends(get_job_service)
):
    try:
        return job_service.list_jobs()
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list jobs."
        )


@router.get(
    "/runtime",
    status_code=status.HTTP_200_OK,
    summary="Inspect Runtime Environment",
    description="Returns complete system hardware diagnostics, GPU state, and versions of AI framework libraries.",
    tags=["GPU Runtime"]
)
async def get_runtime(
    runtime_service: RuntimeService = Depends(get_runtime_service)
):
    try:
        diag = runtime_service.get_diagnostics()
        model_manager = ModelManager()
        scheduler = GPUScheduler()
        
        # Get model details
        model_info = model_manager.get_loaded_model_info()
        
        # Extend with health monitoring statistics
        diag.update({
            "model": model_info.get("loaded_model"),
            "current_model_loaded": model_info.get("loaded_model"),
            "loaded_engine": model_info.get("engine"),
            "engine_version": model_info.get("health", {}).get("version", "None"),
            "engine_health": model_info.get("health", {}).get("initialized", False),
            "model_load_time": model_info.get("health", {}).get("load_time_seconds", 0.0),
            "warmup_status": model_manager._warmup_done,
            "memory_usage": model_info.get("memory", {}),
            "current_queue_length": scheduler.queue_length(),
            "running_job_count": len(scheduler.running_jobs()),
            "gpu_usage": "0%" if not scheduler.running_jobs() else "100%",
            "gpu_temperature": "unknown",
        })
        return diag
    except Exception as e:
        logger.error(f"Error querying runtime: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query system runtime parameters."
        )


@router.get(
    "/gpu",
    status_code=status.HTTP_200_OK,
    summary="Inspect GPU Runtime",
    description="Returns GPU-specific diagnostic settings and memory levels. Alias of /runtime.",
    tags=["GPU Runtime"]
)
async def get_gpu(
    runtime_service: RuntimeService = Depends(get_runtime_service)
):
    return await get_runtime(runtime_service=runtime_service)


@router.get(
    "/models",
    status_code=status.HTTP_200_OK,
    summary="Inspect active loaded models",
    description="Returns information on the currently loaded AI model and engine.",
    tags=["GPU Runtime"]
)
async def get_models():
    model_manager = ModelManager()
    return model_manager.get_loaded_model_info()


@router.post(
    "/jobs/{job_id}/cancel",
    status_code=status.HTTP_200_OK,
    summary="Cancel a running or queued job",
    description="Signals the scheduler and engine to abort the active generation job.",
    tags=["Job Engine"]
)
async def cancel_job(
    job_id: str,
    job_service: JobService = Depends(get_job_service)
):
    try:
        scheduler = GPUScheduler()
        status_data = job_service.get_job_status(job_id)
        
        if status_data["status"] in {"completed", "failed", "cancelled"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job in final state '{status_data['status']}'"
            )
            
        scheduler.cancel(job_id)
        
        if status_data["status"] == "queued":
            job_service.update_job_status(job_id, "cancelled", error_message="Job cancelled by user while queued.")
            scheduler.dequeue(job_id)
            
        return {
            "success": True,
            "job_id": job_id,
            "message": f"Cancellation request sent for job '{job_id}'."
        }
    except BuzzBoxException as bbe:
        raise HTTPException(
            status_code=bbe.status_code,
            detail={"error_code": bbe.error_code, "message": bbe.message}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while cancelling job: {str(e)}"
        )


@router.get(
    "/characters",
    status_code=status.HTTP_200_OK,
    summary="List Characters",
    description="Queries and returns all dynamically discovered characters in storage and their validation statuses.",
    tags=["Characters"]
)
async def list_characters(
    character_service: CharacterService = Depends(get_character_service)
):
    try:
        character_ids = character_service.list_characters()
        characters_list = []
        for char_id in character_ids:
            try:
                char_profile = character_service.load_character(char_id)
                characters_list.append({
                    "id": char_id,
                    "status": char_profile.status
                })
            except Exception:
                characters_list.append({
                    "id": char_id,
                    "status": "error"
                })
        return characters_list
    except Exception as e:
        logger.error(f"Error listing characters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to discover characters."
        )
