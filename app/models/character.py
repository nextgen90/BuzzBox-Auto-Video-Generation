from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class CharacterProfile(BaseModel):
    """
    Pydantic schema representing the full profile of an AI Character,
    including validation metadata, asset paths, and parsed bible content.
    """
    id: str = Field(..., description="Unique identifier for the character, matching the directory name")
    display_name: str = Field(..., description="Human-readable display name of the character")
    version: str = Field("1.0", description="Version of the character configuration")
    default_voice: str = Field(..., description="Default voice synthesis model/profile identifier")
    default_style: str = Field("cinematic", description="Default visual prompt style")
    languages: List[str] = Field(default_factory=list, description="List of supported language codes")
    identity_version: int = Field(1, description="Identity image asset version number")
    voice_version: int = Field(1, description="Voice sample asset version number")
    
    # Asset Paths (Stored as absolute path strings)
    identity_images: List[str] = Field(default_factory=list, description="Absolute paths to identity images (e.g. front.png)")
    poses: List[str] = Field(default_factory=list, description="Absolute paths to standing/sitting visual poses")
    voices: List[str] = Field(default_factory=list, description="Absolute paths to audio voice samples")
    bible_path: str = Field(..., description="Absolute path to the character bible file (.pdf, .md, or .json)")
    bible_content: Optional[str] = Field(None, description="Extracted plain text or JSON content from the character bible")
    
    status: str = Field("ready", description="Operational status of the character (e.g. 'ready', 'draft')")
