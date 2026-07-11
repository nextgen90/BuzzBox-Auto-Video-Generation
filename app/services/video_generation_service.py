import time
import os
import shutil
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from loguru import logger

from app.core.config import settings
from app.engines.video.model_manager import ModelManager
from app.engines.video.generation_result import GenerationResult
from app.services.runtime_service import RuntimeService
from app.services.storage_service import StorageService

# Generation profile configurations
PROFILES = {
    "FAST": {
        "steps": 10,
        "guidance_scale": 4.5,
        "scheduler": "dpm",
        "motion_strength": 3.0
    },
    "STANDARD": {
        "steps": 20,
        "guidance_scale": 6.0,
        "scheduler": "euler",
        "motion_strength": 5.0
    },
    "QUALITY": {
        "steps": 35,
        "guidance_scale": 7.5,
        "scheduler": "euler-ancestral",
        "motion_strength": 7.0
    },
    "ULTRA": {
        "steps": 50,
        "guidance_scale": 9.0,
        "scheduler": "flow",
        "motion_strength": 10.0
    }
}

# Platform default configurations
PLATFORM_DEFAULTS = {
    "youtube": {
        "aspect_ratio": "16:9",
        "resolution": "1080p",
        "fps": 30,
        "subtitles": True
    },
    "instagram": {
        "aspect_ratio": "9:16",
        "resolution": "1080p",
        "fps": 30,
        "subtitles": True
    },
    "tiktok": {
        "aspect_ratio": "9:16",
        "resolution": "720p",
        "fps": 30,
        "subtitles": True
    },
    "facebook": {
        "aspect_ratio": "1:1",
        "resolution": "1080p",
        "fps": 30,
        "subtitles": True
    },
    "future": {
        "aspect_ratio": "16:9",
        "resolution": "1080p",
        "fps": 30,
        "subtitles": True
    }
}

class VideoGenerationService:
    """
    VideoGenerationService handles preparation of video parameters, resolves
    platform/profile configurations, generates the seed, invokes the active engine
    via ModelManager, and manages the compilation of all job output files on disk.
    """
    def __init__(self) -> None:
        self.model_manager = ModelManager()
        self.runtime_service = RuntimeService()
        self.storage_service = StorageService()

    def prepare(self, prompt_data: Dict[str, Any], request_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merges platform defaults, profile values, and explicit overrides to prepare
        the final execution options and reproducible seed.
        """
        # 1. Resolve Platform defaults
        platform = request_params.get("platform", "future").lower()
        if platform not in PLATFORM_DEFAULTS:
            platform = "future"
        defaults = PLATFORM_DEFAULTS[platform]

        # 2. Resolve Profile settings
        profile_name = request_params.get("profile", "STANDARD").upper()
        if profile_name not in PROFILES:
            profile_name = "STANDARD"
        profile_settings = PROFILES[profile_name]

        # 3. Handle explicit overrides or fallback to platform defaults
        aspect_ratio = request_params.get("aspect_ratio") or defaults["aspect_ratio"]
        resolution = request_params.get("resolution") or defaults["resolution"]
        fps = request_params.get("fps") or defaults["fps"]
        duration = request_params.get("duration") or prompt_data.get("duration", 5)

        # 4. Handle Seed generation (Reproducible Seed)
        seed = request_params.get("seed")
        if seed is None or seed <= 0:
            seed = random.randint(1, 2**31 - 1)
        
        # 5. Resolve steps, guidance, scheduler, and motion strength from profile & overrides
        steps = request_params.get("steps") or profile_settings["steps"]
        guidance_scale = request_params.get("guidance_scale") or profile_settings["guidance_scale"]
        scheduler = request_params.get("scheduler") or profile_settings["scheduler"]
        motion_strength = request_params.get("motion_strength") or profile_settings["motion_strength"]

        # Formulate prepared parameters payload
        prepared = {
            "prompt_instruction": prompt_data.get("prompt_instruction", ""),
            "negative_prompt": request_params.get("negative_prompt") or prompt_data.get("negative_prompt", ""),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "fps": fps,
            "duration": duration,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "scheduler": scheduler,
            "motion_strength": motion_strength,
            "platform": platform,
            "profile": profile_name,
            "character": prompt_data.get("character"),
            "language": prompt_data.get("language")
        }

        # Enhance the prompt instruction slightly with platform framing cues
        if platform == "tiktok" or platform == "instagram":
            prepared["prompt_instruction"] += " Vertical video format optimized for short-form mobile content."
        elif platform == "youtube":
            prepared["prompt_instruction"] += " Landscape cinematic video format optimized for television/desktop screens."

        logger.info(f"Prepared generation parameters: Platform={platform}, Profile={profile_name}, Seed={seed}, Size={resolution}")
        return prepared

    def generate(
        self, 
        prepared_params: Dict[str, Any], 
        callback: Optional[Callable[[int, int, Any], None]] = None
    ) -> GenerationResult:
        """Invokes the active engine to generate the video."""
        engine = self.model_manager.get_engine()
        logger.info(f"Dispatching generation to engine '{engine.name()}'")
        return engine.generate(prepared_params, callback=callback)

    def save(
        self, 
        job_id: str, 
        result: GenerationResult, 
        request_payload: Dict[str, Any], 
        status_payload: Dict[str, Any], 
        prompt_data: Dict[str, Any], 
        logs_content: str
    ) -> Path:
        """
        Copies output files to outputs/{job_id}/ and generates all metadata,
        runtime diagnostics, generation statistics JSON files, and copies logs.
        Also replicates these files inside storage/jobs/{job_id}/ for backward compatibility.
        """
        # Create output directories
        output_job_dir = settings.OUTPUT_DIR / job_id
        output_job_dir.mkdir(parents=True, exist_ok=True)

        storage_job_dir = settings.JOBS_DIR / job_id
        storage_job_dir.mkdir(parents=True, exist_ok=True)

        # 1. Define target paths in outputs/job_id/
        target_video = output_job_dir / "video.mp4"
        target_thumb = output_job_dir / "thumbnail.jpg"
        target_preview = output_job_dir / "preview.gif"

        # Copy generated files from temp to outputs/
        if os.path.exists(result.video_path):
            shutil.copy2(result.video_path, target_video)
        if os.path.exists(result.thumbnail_path):
            shutil.copy2(result.thumbnail_path, target_thumb)
        if os.path.exists(result.preview_path):
            shutil.copy2(result.preview_path, target_preview)

        # 2. Gather diagnostics and build runtime.json
        diag = self.runtime_service.get_diagnostics()
        runtime_payload = {
            "gpu": diag.get("gpu"),
            "cuda": diag.get("cuda"),
            "vram_total": diag.get("vram"),
            "torch_version": diag.get("torch"),
            "diffusers_version": diag.get("diffusers"),
            "transformers_version": diag.get("transformers"),
            "allocated_vram": result.statistics.get("memory_allocated"),
            "peak_vram": result.statistics.get("memory_peak"),
            "device": result.metadata.get("device"),
            "precision": result.metadata.get("precision")
        }

        # 3. Build generation.json
        generation_payload = {
            "job_id": job_id,
            "engine": result.engine_name,
            "engine_version": result.engine_version,
            "steps": result.metadata.get("steps"),
            "guidance_scale": result.metadata.get("guidance_scale"),
            "seed": result.seed,
            "generation_time_seconds": result.generation_time,
            "fps": result.fps,
            "duration": result.duration,
            "resolution": result.resolution,
            "fallback_mode": result.metadata.get("fallback_mode", False),
            "statistics": result.statistics,
            "warnings": result.warnings,
            "errors": result.errors
        }

        # 4. Build metadata.json
        metadata_payload = {
            "generation_time": result.generation_time,
            "FPS": result.fps,
            "resolution": result.resolution,
            "aspect_ratio": result.aspect_ratio,
            "model": settings.MODEL_NAME,
            "seed": result.seed,
            "steps": result.metadata.get("steps"),
            "scheduler": result.metadata.get("scheduler", "default"),
            "GPU": diag.get("gpu"),
            "VRAM": diag.get("vram"),
            "prompt": result.metadata.get("prompt_instruction", request_payload.get("script", "")),
            "negative_prompt": result.metadata.get("negative_prompt", ""),
            "output_path": str(target_video.resolve()),
            "S3_URL": status_payload.get("video_url", "")
        }

        # Save JSON files to outputs/job_id/
        self.storage_service.save_json(output_job_dir / "request.json", request_payload)
        self.storage_service.save_json(output_job_dir / "status.json", status_payload)
        self.storage_service.save_json(output_job_dir / "prompt.json", prompt_data)
        self.storage_service.save_json(output_job_dir / "runtime.json", runtime_payload)
        self.storage_service.save_json(output_job_dir / "generation.json", generation_payload)
        self.storage_service.save_json(output_job_dir / "metadata.json", metadata_payload)

        # Write logs.txt to outputs/job_id/
        with open(output_job_dir / "logs.txt", "w", encoding="utf-8") as f:
            f.write(logs_content)

        # Replicate all files inside storage/jobs/{job_id}/ for backward compatibility
        shutil.copy2(target_video, storage_job_dir / "video.mp4")
        shutil.copy2(target_thumb, storage_job_dir / "thumbnail.jpg")
        shutil.copy2(target_preview, storage_job_dir / "preview.gif")
        
        self.storage_service.save_json(storage_job_dir / "request.json", request_payload)
        self.storage_service.save_json(storage_job_dir / "status.json", status_payload)
        self.storage_service.save_json(storage_job_dir / "prompt.json", prompt_data)
        self.storage_service.save_json(storage_job_dir / "runtime.json", runtime_payload)
        self.storage_service.save_json(storage_job_dir / "generation.json", generation_payload)
        self.storage_service.save_json(storage_job_dir / "metadata.json", metadata_payload)
        
        with open(storage_job_dir / "logs.txt", "w", encoding="utf-8") as f:
            f.write(logs_content)

        logger.info(f"Saved complete job artifacts for job {job_id} to outputs/ and storage/jobs/")
        return output_job_dir
