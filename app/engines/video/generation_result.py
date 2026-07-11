from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class GenerationResult(BaseModel):
    """
    Standardized result model returned by all video engines.
    Ensures that the calling service consumes a uniform payload regardless of engine.
    """
    video_path: str = Field(..., description="Absolute filesystem path to the generated MP4 file")
    thumbnail_path: str = Field(..., description="Absolute filesystem path to the generated JPEG thumbnail")
    preview_path: str = Field(..., description="Absolute filesystem path to the generated animated preview GIF")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary associated with generation")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Inference performance statistics (VRAM, timing)")
    seed: int = Field(..., description="Seed used for reproducible generation")
    fps: int = Field(..., description="Frames per second of the output video")
    duration: int = Field(..., description="Duration of the output video in seconds")
    resolution: str = Field(..., description="Target resolution string (e.g. 1920x1080)")
    aspect_ratio: str = Field(..., description="Aspect ratio used")
    generation_time: float = Field(..., description="Total time taken to generate in seconds")
    engine_name: str = Field(..., description="Name of the engine used")
    engine_version: str = Field(..., description="Version of the engine used")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings during generation")
    errors: List[str] = Field(default_factory=list, description="Encountered non-fatal errors")
