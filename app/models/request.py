from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator

class VideoGenerationRequest(BaseModel):
    """
    Schema representing video generation parameters, with strict validation
    rules to guarantee clean and schema-compliant data ingestion.
    """
    request_id: Optional[str] = Field(
        default=None, 
        description="Optional client-defined unique request identifier to trace requests"
    )
    character: str = Field(
        ..., 
        min_length=1, 
        max_length=100, 
        description="Name or short descriptor of the main visual character profile"
    )
    language: str = Field(
        ..., 
        min_length=2, 
        max_length=50, 
        description="Target language code or name for script voiceover (e.g. 'en-US', 'es', 'english')"
    )
    script: str = Field(
        ..., 
        min_length=1, 
        max_length=5000, 
        description="Text script containing dialogues or cues for voice synthesis and visual matching"
    )
    duration: int = Field(
        ..., 
        gt=0, 
        le=3600, 
        description="Target video duration in seconds (must be greater than 0 and less than or equal to 3600)"
    )
    aspect_ratio: Optional[str] = Field(
        default=None, 
        description="Target aspect ratio. Supported values: '16:9', '9:16', '1:1', '4:5'"
    )
    resolution: Optional[str] = Field(
        default=None, 
        description="Target resolution. Supported values: '720p', '1080p', '4k' or standard dimensions (e.g., '1920x1080')"
    )
    fps: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=120, 
        description="Target frames per second (must be between 1 and 120)"
    )
    background_music: bool = Field(
        default=False, 
        description="Whether to incorporate a background music track"
    )
    subtitles: bool = Field(
        default=True, 
        description="Whether to generate and overlay visible subtitles"
    )
    camera: Optional[str] = Field(
        default=None, 
        description="Optional camera framing instruction (e.g. 'medium shot, static camera')"
    )
    style: Optional[str] = Field(
        default=None, 
        description="Optional prompt style descriptor (e.g. 'cinematic', 'anime')"
    )
    negative_prompt: Optional[str] = Field(
        default=None, 
        description="Optional negative prompt keywords to prevent visual defects"
    )
    platform: Optional[str] = Field(
        default="future",
        description="Target social platform (e.g. 'youtube', 'instagram', 'facebook', 'tiktok', 'future')"
    )
    profile: Optional[str] = Field(
        default="STANDARD",
        description="Rendering quality profile (e.g. 'FAST', 'STANDARD', 'QUALITY', 'ULTRA')"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional seed for deterministic reproducibility"
    )
    width: Optional[int] = Field(
        default=None,
        description="Optional target width override"
    )
    height: Optional[int] = Field(
        default=None,
        description="Optional target height override"
    )
    steps: Optional[int] = Field(
        default=None,
        description="Optional custom steps override"
    )
    guidance_scale: Optional[float] = Field(
        default=None,
        description="Optional custom guidance scale override"
    )
    scheduler: Optional[str] = Field(
        default=None,
        description="Optional custom scheduler override"
    )
    motion_strength: Optional[float] = Field(
        default=None,
        description="Optional custom motion strength override"
    )
    voice: Optional[str] = Field(
        default=None,
        description="Optional custom voice profile override"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Key-value dictionary containing custom operational metadata"
    )

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, val: Optional[str]) -> Optional[str]:
        """Enforces a predefined set of professional aspect ratios."""
        if val is None:
            return val
        allowed_aspects = {"16:9", "9:16", "1:1", "4:5"}
        if val not in allowed_aspects:
            raise ValueError(
                f"Aspect ratio '{val}' is invalid. Supported aspect ratios: {sorted(allowed_aspects)}"
            )
        return val

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, val: Optional[str]) -> Optional[str]:
        """Validates that the target resolution conforms to supported formats."""
        if val is None:
            return val
        normalized = val.strip().lower()
        allowed_resolutions = {
            "720p", "1080p", "4k", 
            "1280x720", "1920x1080", "3840x2160"
        }
        if normalized not in allowed_resolutions:
            raise ValueError(
                f"Resolution '{val}' is invalid. Supported resolutions: {sorted(allowed_resolutions)}"
            )
        return normalized

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "req-123456",
                "character": "Cyberpunk Nomad",
                "language": "en-US",
                "script": "Welcome to the future. The neon lights illuminate the dark alleyways...",
                "duration": 60,
                "aspect_ratio": "16:9",
                "resolution": "1080p",
                "fps": 30,
                "background_music": True,
                "subtitles": True,
                "metadata": {
                    "client": "web-app",
                    "priority": "high"
                }
            }
        }
    }
