from pydantic import BaseModel, Field

class VideoGenerationResponse(BaseModel):
    """
    Response schema returning essential metadata about a newly created
    video generation job, ensuring clients receive standardized state info.
    """
    success: bool = Field(
        ..., 
        description="Indicates whether the video generation job was successfully accepted and queued"
    )
    job_id: str = Field(
        ..., 
        description="The unique UUID string associated with the video generation job"
    )
    status: str = Field(
        ..., 
        description="Current operational status of the job (e.g. 'queued')"
    )
    message: str = Field(
        ..., 
        description="Human-readable message regarding the status or ingestion step"
    )
    estimated_time: int = Field(
        ..., 
        description="Estimated processing time in seconds based on length and resolution complexity"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "job_id": "8b51d8b7-6fa3-43f1-9ee5-a7b29a25b74b",
                "status": "queued",
                "message": "Video generation request accepted.",
                "estimated_time": 120
            }
        }
    }
