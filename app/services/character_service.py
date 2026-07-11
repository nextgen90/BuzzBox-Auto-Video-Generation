import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger
from app.core.config import settings
from app.core.exceptions import (
    CharacterMissingError,
    BibleMissingError,
    IdentityMissingError,
    StorageError
)
from app.models.character import CharacterProfile

class CharacterService:
    """
    CharacterService manages AI Character life-cycle, including dynamic discovery,
    asset validation, metadata ingestion, and character bible parsing.
    """
    def __init__(self) -> None:
        self.characters_dir: Path = settings.CHARACTERS_DIR
        self.characters_dir.mkdir(parents=True, exist_ok=True)

    def list_characters(self) -> List[str]:
        """Discovers and returns list of character IDs based on directory names."""
        try:
            return [d.name for d in self.characters_dir.iterdir() if d.is_dir()]
        except Exception as e:
            logger.error(f"Failed to list characters: {str(e)}")
            raise StorageError(f"Failed to scan characters storage directory: {str(e)}")

    def validate_character(self, character_id: str) -> None:
        """
        Validates character directory existence and required assets.
        Raises specific domain exceptions if files are missing or malformed.
        """
        char_path = self.characters_dir / character_id
        if not char_path.is_dir():
            raise CharacterMissingError(f"Character '{character_id}' does not exist in storage.")

        metadata_file = char_path / "metadata.json"
        if not metadata_file.is_file():
            raise CharacterMissingError(f"Character '{character_id}' is missing required metadata.json file.")

        # Try parsing metadata.json
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                json.load(f)
        except Exception as e:
            raise StorageError(f"Character '{character_id}' has corrupt or unreadable metadata.json: {str(e)}")

        # Validate Identity folder and files
        identity_dir = char_path / "identity"
        if not identity_dir.is_dir():
            raise IdentityMissingError(f"Character '{character_id}' is missing required 'identity' folder.")
        
        identity_files = [
            f for f in identity_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if not identity_files:
            raise IdentityMissingError(f"Character '{character_id}' has an empty or invalid 'identity' folder.")

        # Validate Poses folder and files
        poses_dir = char_path / "poses"
        if not poses_dir.is_dir():
            raise IdentityMissingError(f"Character '{character_id}' is missing required 'poses' folder.")
        
        pose_files = [
            f for f in poses_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if not pose_files:
            raise IdentityMissingError(f"Character '{character_id}' has an empty or invalid 'poses' folder.")

        # Validate Voices folder and files
        voices_dir = char_path / "voices"
        if not voices_dir.is_dir():
            raise IdentityMissingError(f"Character '{character_id}' is missing required 'voices' folder.")
        
        voice_files = [
            f for f in voices_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".wav", ".mp3", ".m4a"}
        ]
        if not voice_files:
            raise IdentityMissingError(f"Character '{character_id}' has an empty or invalid 'voices' folder.")

        # Validate Bible folder and files
        bible_dir = char_path / "bible"
        if not bible_dir.is_dir():
            raise BibleMissingError(f"Character '{character_id}' is missing required 'bible' folder.")
        
        bible_files = [
            f for f in bible_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".pdf", ".md", ".json"}
        ]
        if not bible_files:
            raise BibleMissingError(f"Character '{character_id}' has no valid character bible file (.pdf, .md, or .json) in 'bible' folder.")

    def get_character_metadata(self, character_id: str) -> Dict[str, Any]:
        """Loads and returns character metadata JSON dictionary."""
        self.validate_character(character_id)
        metadata_file = self.characters_dir / character_id / "metadata.json"
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise StorageError(f"Failed to read metadata for character '{character_id}': {str(e)}")

    def get_character_assets(self, character_id: str) -> Dict[str, List[str]]:
        """Returns absolute path strings for all visual, voice, and bible assets."""
        self.validate_character(character_id)
        char_path = self.characters_dir / character_id
        
        identity_paths = [str(f.resolve()) for f in (char_path / "identity").iterdir() if f.is_file()]
        pose_paths = [str(f.resolve()) for f in (char_path / "poses").iterdir() if f.is_file()]
        voice_paths = [str(f.resolve()) for f in (char_path / "voices").iterdir() if f.is_file()]
        bible_paths = [str(f.resolve()) for f in (char_path / "bible").iterdir() if f.is_file()]

        return {
            "identity_images": sorted(identity_paths),
            "poses": sorted(pose_paths),
            "voices": sorted(voice_paths),
            "bible_path": sorted(bible_paths)
        }

    def load_character(self, character_id: str) -> CharacterProfile:
        """Loads character metadata and assets into a Pydantic CharacterProfile model."""
        self.validate_character(character_id)
        char_path = self.characters_dir / character_id
        
        metadata = self.get_character_metadata(character_id)
        assets = self.get_character_assets(character_id)
        
        # Load Bible Content
        bible_path = Path(assets["bible_path"][0])
        bible_content: Optional[str] = None
        
        try:
            suffix = bible_path.suffix.lower()
            if suffix == ".json":
                with open(bible_path, "r", encoding="utf-8") as f:
                    bible_content = json.dumps(json.load(f))
            elif suffix == ".md":
                with open(bible_path, "r", encoding="utf-8") as f:
                    bible_content = f.read()
            elif suffix == ".pdf":
                bible_content = f"[PDF Reference: {bible_path.name} | Size: {bible_path.stat().st_size} bytes]"
        except Exception as e:
            logger.warning(f"Could not parse bible content at {bible_path}: {str(e)}")
            bible_content = f"[Unreadable Bible: {bible_path.name}]"

        # Create CharacterProfile
        return CharacterProfile(
            id=character_id,
            display_name=metadata.get("display_name", character_id.capitalize()),
            version=metadata.get("version", "1.0"),
            default_voice=metadata.get("default_voice", f"{character_id}_voice"),
            default_style=metadata.get("default_style", "cinematic"),
            languages=metadata.get("languages", ["en"]),
            identity_version=metadata.get("identity_version", 1),
            voice_version=metadata.get("voice_version", 1),
            identity_images=assets["identity_images"],
            poses=assets["poses"],
            voices=assets["voices"],
            bible_path=str(bible_path.resolve()),
            bible_content=bible_content,
            status=metadata.get("status", "ready")
        )
