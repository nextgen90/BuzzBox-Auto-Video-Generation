import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger
from app.core.exceptions import JobNotFoundError, InvalidJobError
from app.services.storage_service import StorageService

# Job states, progress percentage, step descriptions
JOB_STATES_METRIC = {
    "queued": {
        "progress": 5,
        "step": "Job queued for processing"
    },
    "validating": {
        "progress": 15,
        "step": "Validating request parameters"
    },
    "validating_request": {
        "progress": 15,
        "step": "Validating request parameters"
    },
    "loading_character": {
        "progress": 25,
        "step": "Loading character assets"
    },
    "building_prompt": {
        "progress": 40,
        "step": "Building AI prompt context"
    },
    "loading_model": {
        "progress": 60,
        "step": "Loading AI models into memory"
    },
    "preparing_inputs": {
        "progress": 80,
        "step": "Preparing video parameters and seed"
    },
    "runtime_check": {
        "progress": 80,
        "step": "Verifying system runtime resources"
    },
    "waiting_gpu": {
        "progress": 80,
        "step": "Waiting for GPU resource allocation"
    },
    "generating": {
        "progress": 90,
        "step": "Generating video via AI model"
    },
    "encoding": {
        "progress": 95,
        "step": "Encoding frames to MP4 and GIF"
    },
    "uploading": {
        "progress": 98,
        "step": "Uploading complete job artifacts to storage"
    },
    "completed": {
        "progress": 100,
        "step": "Generation completed successfully"
    },
    "failed": {
        "progress": 100,
        "step": "Generation failed"
    },
    "cancelled": {
        "progress": 100,
        "step": "Job cancelled by user"
    }
}

class JobService:
    """
    JobService handles creation, updates, querying, listing, and state machine
    transitions of generative AI jobs, persisting status records under storage/jobs/
    """
    def __init__(self, storage_service: Optional[StorageService] = None) -> None:
        self.storage_service = storage_service or StorageService()

    def generate_job_id(self) -> str:
        """Generates a secure UUID v4 string representing the job."""
        return str(uuid.uuid4())

    def create_job(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates and registers a new job, preparing its on-disk subdirectory and
        initializing request.json, status.json, metadata.json, and timestamps.json.
        """
        job_id = self.generate_job_id()
        self.storage_service.create_job_directory(job_id)
        
        # Calculate estimated total time (2 seconds per video second + base setup overhead of 10s)
        duration = request_payload.get("duration", 30)
        estimated_time = (duration * 2) + 10

        now_str = datetime.now(timezone.utc).isoformat()
        
        # 1. request.json
        req_path = self.storage_service.jobs_dir / job_id / "request.json"
        self.storage_service.save_json(req_path, request_payload)

        # 2. metadata.json
        meta_path = self.storage_service.jobs_dir / job_id / "metadata.json"
        metadata_payload = {
            "aspect_ratio": request_payload.get("aspect_ratio"),
            "resolution": request_payload.get("resolution"),
            "fps": request_payload.get("fps"),
            "background_music": request_payload.get("background_music", False),
            "subtitles": request_payload.get("subtitles", True),
            "request_id": request_payload.get("request_id") or job_id,
            "estimated_time": estimated_time,
            "camera": request_payload.get("camera", "medium shot, static camera, eye level"),
            "style": request_payload.get("style", "cinematic"),
            "negative_prompt": request_payload.get("negative_prompt", ""),
            "custom_metadata": request_payload.get("metadata", {})
        }
        self.storage_service.save_json(meta_path, metadata_payload)

        # 3. timestamps.json
        time_path = self.storage_service.jobs_dir / job_id / "timestamps.json"
        timestamps = {
            "created_at": now_str,
            "updated_at": now_str,
            "started_at": None,
            "completed_at": None,
            "failed_at": None
        }
        self.storage_service.save_json(time_path, timestamps)

        # 4. status.json
        status_path = self.storage_service.jobs_dir / job_id / "status.json"
        status_payload = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "current_step": "Job queued for processing",
            "estimated_remaining_seconds": estimated_time,
            "error": None
        }
        self.storage_service.save_json(status_path, status_payload)

        # Append initial log
        self.storage_service.append_job_log(job_id, "queued", f"Job initialized. Estimated execution time: {estimated_time}s")

        return {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "created_at": now_str,
            "updated_at": now_str
        }

    def update_job_status(self, job_id: str, status: str, error_message: Optional[str] = None) -> None:
        """
        Updates the job state, progress percentage, timestamps, remaining duration,
        and logs the transition to the job's logs.txt.
        """
        job_dir = self.storage_service.jobs_dir / job_id
        if not job_dir.is_dir():
            raise JobNotFoundError(f"Job '{job_id}' does not exist.")

        if status not in JOB_STATES_METRIC:
            raise InvalidJobError(f"Job status '{status}' is invalid.")

        now_str = datetime.now(timezone.utc).isoformat()
        
        # Load files
        status_path = job_dir / "status.json"
        meta_path = job_dir / "metadata.json"
        time_path = job_dir / "timestamps.json"
        
        status_data = self.storage_service.load_json(status_path)
        meta_data = self.storage_service.load_json(meta_path)
        time_data = self.storage_service.load_json(time_path)

        # Calculate remaining time
        estimated_time = meta_data.get("estimated_time", 60)
        metrics = JOB_STATES_METRIC[status]
        progress = metrics["progress"]
        step = metrics["step"]
        
        remaining = max(0, int(estimated_time * (1 - (progress / 100.0))))
        if status in {"completed", "failed"}:
            remaining = 0

        # Update timestamps
        time_data["updated_at"] = now_str
        if status == "validating_request" and not time_data.get("started_at"):
            time_data["started_at"] = now_str
        elif status == "completed":
            time_data["completed_at"] = now_str
        elif status == "failed":
            time_data["failed_at"] = now_str

        # Update status
        status_data["status"] = status
        status_data["progress"] = progress
        status_data["current_step"] = step
        status_data["estimated_remaining_seconds"] = remaining
        if error_message:
            status_data["error"] = error_message
            step = f"Error: {error_message}"

        # Save updates
        self.storage_service.save_json(status_path, status_data)
        self.storage_service.save_json(time_path, time_data)

        # Log transition
        self.storage_service.append_job_log(job_id, status, step)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Fetches and maps a job's progress status from disk storage, enriching it with statistics and URLs."""
        job_dir = self.storage_service.jobs_dir / job_id
        if not job_dir.is_dir():
            raise JobNotFoundError(f"Job '{job_id}' does not exist.")

        status_path = job_dir / "status.json"
        time_path = job_dir / "timestamps.json"
        meta_path = job_dir / "metadata.json"
        gen_path = job_dir / "generation.json"
        
        status_data = self.storage_service.load_json(status_path)
        time_data = self.storage_service.load_json(time_path)

        # Retrieve optional/enriched fields
        estimated_time = 0
        video_url = status_data.get("video_url")
        thumbnail_url = status_data.get("thumbnail_url")
        generation_statistics = {}

        if meta_path.is_file():
            try:
                meta_data = self.storage_service.load_json(meta_path)
                estimated_time = meta_data.get("estimated_time", 0)
                if not video_url:
                    video_url = meta_data.get("S3_URL")
                if not thumbnail_url:
                    # In mock or standard uploads, construct path
                    thumbnail_url = meta_data.get("thumbnail_url")
            except Exception:
                pass

        if gen_path.is_file():
            try:
                gen_data = self.storage_service.load_json(gen_path)
                generation_statistics = gen_data.get("statistics") or {}
            except Exception:
                pass

        return {
            "job_id": status_data["job_id"],
            "status": status_data["status"],
            "progress": status_data["progress"],
            "current_step": status_data["current_step"],
            "estimated_remaining_seconds": status_data["estimated_remaining_seconds"],
            "estimated_time": estimated_time,
            "video_url": video_url,
            "thumbnail_url": thumbnail_url,
            "generation_statistics": generation_statistics,
            "error": status_data.get("error"),
            "created_at": time_data["created_at"],
            "updated_at": time_data["updated_at"]
        }

    def list_jobs(self) -> List[Dict[str, Any]]:
        """Queries and lists all jobs in storage, ordered newest first based on created_at."""
        jobs_list = []
        try:
            job_dirs = [d for d in self.storage_service.jobs_dir.iterdir() if d.is_dir()]
        except Exception as e:
            logger.error(f"Failed listing jobs directory: {str(e)}")
            return []

        for job_dir in job_dirs:
            job_id = job_dir.name
            try:
                status_path = job_dir / "status.json"
                time_path = job_dir / "timestamps.json"
                
                if status_path.is_file() and time_path.is_file():
                    status_data = self.storage_service.load_json(status_path)
                    time_data = self.storage_service.load_json(time_path)
                    jobs_list.append({
                        "job_id": job_id,
                        "status": status_data["status"],
                        "progress": status_data["progress"],
                        "current_step": status_data["current_step"],
                        "estimated_remaining_seconds": status_data["estimated_remaining_seconds"],
                        "error": status_data.get("error"),
                        "created_at": time_data["created_at"],
                        "updated_at": time_data["updated_at"]
                    })
            except Exception as e:
                logger.warning(f"Error loading job {job_id} details: {str(e)}")
                continue

        # Sort newest first (ISO 8601 string sort works perfectly for timestamps)
        jobs_list.sort(key=lambda x: x["created_at"], reverse=True)
        return jobs_list
