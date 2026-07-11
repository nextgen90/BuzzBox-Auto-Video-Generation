import json
from pathlib import Path
from typing import Any, Dict
from loguru import logger
from app.core.config import settings
from app.core.exceptions import StorageError
from app.models.character import CharacterProfile

class PromptService:
    """
    PromptService is responsible for orchestrating the generation of structured
    AI prompt context payloads, merging characters, bibles, and input scripts.
    It saves debugging outputs to the configured prompts storage hierarchy.
    """
    def __init__(self) -> None:
        self.prompts_dir: Path = settings.PROMPTS_DIR
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def merge_character_context(self, script: str, character: CharacterProfile, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combines character identity, voice references, and custom settings
        to formulate a comprehensive rendering runtime configuration.
        """
        # Read the defaults from the character if not overridden in options
        style = options.get("style") or character.default_style
        camera = options.get("camera") or "medium shot, static camera, eye level"
        negative_prompt = options.get("negative_prompt") or "blurry, low quality, distorted, extra limbs"
        language = options.get("language") or (character.languages[0] if character.languages else "en")
        
        # Compile a visual/narrative prompt instruction
        prompt_instruction = (
            f"Synthesize voiceover in {language} for script: '{script}'. "
            f"Generate video with character '{character.display_name}' in '{style}' style. "
            f"Camera framing: '{camera}'."
        )

        return {
            "script": script,
            "character": character.id,
            "language": language,
            "duration": options.get("duration", 30),
            "camera": camera,
            "style": style,
            "negative_prompt": negative_prompt,
            "identity_reference": character.identity_images,
            "pose_reference": character.poses,
            "voice_reference": character.voices,
            "bible_reference": character.bible_path,
            "bible_content": character.bible_content,
            "character_metadata": {
                "id": character.id,
                "display_name": character.display_name,
                "version": character.version,
                "default_voice": character.default_voice,
                "default_style": character.default_style,
                "languages": character.languages,
                "identity_version": character.identity_version,
                "voice_version": character.voice_version
            },
            "prompt_instruction": prompt_instruction
        }

    def build_prompt(self, script: str, character: CharacterProfile, options: Dict[str, Any]) -> Dict[str, Any]:
        """Formulates the final merged AI prompt dictionary."""
        return self.merge_character_context(script, character, options)

    def export_prompt_json(self, job_id: str, prompt_data: Dict[str, Any]) -> Path:
        """
        Exports prompt artifacts to storage/prompts/job_{job_id}/ directory.
        Writes raw_prompt.json, compiled_prompt.json, and negative_prompt.txt.
        """
        job_prompt_dir = self.prompts_dir / f"job_{job_id}"
        try:
            job_prompt_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. raw_prompt.json
            raw_prompt_path = job_prompt_dir / "raw_prompt.json"
            raw_data = {
                "script": prompt_data["script"],
                "character": prompt_data["character"],
                "language": prompt_data["language"],
                "duration": prompt_data["duration"],
                "camera": prompt_data["camera"],
                "style": prompt_data["style"],
                "negative_prompt": prompt_data["negative_prompt"]
            }
            with open(raw_prompt_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=4)

            # 2. compiled_prompt.json
            compiled_prompt_path = job_prompt_dir / "compiled_prompt.json"
            with open(compiled_prompt_path, "w", encoding="utf-8") as f:
                json.dump(prompt_data, f, indent=4)

            # 3. negative_prompt.txt
            negative_prompt_path = job_prompt_dir / "negative_prompt.txt"
            with open(negative_prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt_data["negative_prompt"])

            logger.info(f"Successfully exported prompts for job {job_id} to {job_prompt_dir}")
            return job_prompt_dir

        except Exception as e:
            logger.error(f"Failed to export prompt files for job {job_id}: {str(e)}")
            raise StorageError(f"Failed writing prompt files for job {job_id}: {str(e)}")
