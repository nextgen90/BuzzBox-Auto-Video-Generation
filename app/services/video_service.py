from typing import Any, Dict
from loguru import logger
from app.models.request import VideoGenerationRequest
from app.models.response import VideoGenerationResponse
from app.services.job_service import JobService

class VideoService:
    """
    VideoService acts as a backward-compatibility layer pointing to the new,
    generic JobService. This ensures Sprint 1 API dependencies remain unbroken.
    """
    def __init__(self) -> None:
        self.job_service = JobService()

    def validate_request(self, request: VideoGenerationRequest) -> None:
        """Runs validation checks on the request."""
        if request.duration > 1800:
            raise ValueError(
                f"Requested video duration of {request.duration}s exceeds the maximum allowed length of 1800s (30 minutes)."
            )
        if "priority" in request.metadata:
            priority = str(request.metadata["priority"]).lower()
            if priority not in {"low", "medium", "high"}:
                raise ValueError("If 'priority' is set in metadata, it must be 'low', 'medium', or 'high'.")

    def create_job(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        """
        Adapts VideoGenerationRequest input and delegates job creation
        to JobService, returning a VideoGenerationResponse.
        """
        self.validate_request(request)
        
        # Serialize the Pydantic request to dictionary for the JobService
        request_dict = request.model_dump()
        
        # Delegate job creation
        job_info = self.job_service.create_job(request_dict)
        
        # Estimate processing time (2 seconds per video second + base setup overhead of 10s)
        estimated_time = (request.duration * 2) + 10

        return VideoGenerationResponse(
            success=True,
            job_id=job_info["job_id"],
            status=job_info["status"],
            message="Video generation request accepted.",
            estimated_time=estimated_time
        )
