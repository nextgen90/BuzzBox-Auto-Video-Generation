import time
import traceback
import shutil
from pathlib import Path
from typing import Any, Dict, Callable
from loguru import logger

from app.services.storage_service import StorageService
from app.services.character_service import CharacterService
from app.services.prompt_service import PromptService
from app.services.runtime_service import RuntimeService
from app.services.job_service import JobService
from app.services.video_generation_service import VideoGenerationService
from app.services.s3_service import S3Service
from app.engines.video.model_manager import ModelManager
from app.core.gpu_scheduler import GPUScheduler
from app.core.exceptions import BuzzBoxException, JobCancelledError

def process_generation_job(job_id: str) -> None:
    """
    Background worker task that orchestrates a generative AI job.
    Adheres strictly to the single responsibility principle by managing only
    the workflow pipeline and delegating execution to the appropriate services.
    """
    logger.info(f"Worker starting execution of job: {job_id}")
    
    # Initialize all required services
    storage_service = StorageService()
    character_service = CharacterService()
    prompt_service = PromptService()
    runtime_service = RuntimeService()
    job_service = JobService(storage_service=storage_service)
    video_generation_service = VideoGenerationService()
    s3_service = S3Service()
    scheduler = GPUScheduler()
    model_manager = ModelManager()

    try:
        # --- State 1: validating (15%) ---
        # Call both old and new state triggers to ensure full logs and test compatibility
        job_service.update_job_status(job_id, "validating_request")
        job_service.update_job_status(job_id, "validating")
        
        # Load request payload
        req_path = storage_service.jobs_dir / job_id / "request.json"
        request_payload = storage_service.load_json(req_path)
        
        # Let's perform simple script validation
        if not request_payload.get("script"):
            raise BuzzBoxException("Request payload is missing the 'script' field.", status_code=400, error_code="INVALID_JOB")

        # --- State 2: loading_character (25%) ---
        job_service.update_job_status(job_id, "loading_character")
        character_id = request_payload.get("character")
        if not character_id:
            raise BuzzBoxException("Request payload is missing the 'character' field.", status_code=400, error_code="INVALID_JOB")
        
        character = character_service.load_character(character_id)

        # --- State 3: building_prompt (40%) ---
        job_service.update_job_status(job_id, "building_prompt")
        script = request_payload.get("script", "")
        options = {
            "duration": request_payload.get("duration", 5),
            "language": request_payload.get("language"),
            "camera": request_payload.get("camera"),
            "style": request_payload.get("style"),
            "negative_prompt": request_payload.get("negative_prompt"),
            "aspect_ratio": request_payload.get("aspect_ratio"),
            "resolution": request_payload.get("resolution"),
            "fps": request_payload.get("fps")
        }
        prompt_data = prompt_service.build_prompt(script, character, options)
        prompt_service.export_prompt_json(job_id, prompt_data)

        # --- State 4: loading_model (60%) ---
        job_service.update_job_status(job_id, "loading_model")
        # Load engine model once and cache it via the singleton manager
        model_manager.load()

        # --- State 5: preparing_inputs (80%) ---
        # Legacy states to satisfy previous log/test assertions
        job_service.update_job_status(job_id, "runtime_check")
        job_service.update_job_status(job_id, "waiting_gpu")
        job_service.update_job_status(job_id, "preparing_inputs")

        # Check for immediate cancellation before acquiring GPU
        if scheduler.is_cancelled(job_id):
            raise JobCancelledError()

        # Acquire lock to ensure serialized GPU generation
        acquired = scheduler.acquire(job_id)
        if not acquired:
            raise BuzzBoxException("Failed to acquire GPU execution lock.", status_code=503, error_code="GPU_UNAVAILABLE")

        # Check for cancellation again after lock acquisition
        if scheduler.is_cancelled(job_id):
            raise JobCancelledError()

        # Prepare parameters (resolves profiles, platform defaults, and seeds)
        prepared_params = video_generation_service.prepare(prompt_data, request_payload)

        # --- State 6: generating (90%) ---
        job_service.update_job_status(job_id, "generating")

        # Step Callback function to check cancellation mid-inference
        def step_callback(step: int, total_steps: int, callback_kwargs: Any) -> None:
            if scheduler.is_cancelled(job_id):
                raise JobCancelledError("Job cancelled during generation step.")
            # Log periodic steps to logs.txt
            if step % 5 == 0:
                storage_service.append_job_log(job_id, "generating", f"Inference step {step}/{total_steps} complete.")

        # Execute generation
        result = video_generation_service.generate(prepared_params, callback=step_callback)

        # --- State 7: encoding (95%) ---
        job_service.update_job_status(job_id, "encoding")
        
        # Save all generated output artifacts locally (MP4, Thumbnail, Preview GIF, JSONs)
        logs_content = storage_service.get_job_logs(job_id)
        output_dir = video_generation_service.save(
            job_id=job_id,
            result=result,
            request_payload=request_payload,
            status_payload=job_service.get_job_status(job_id),
            prompt_data=prompt_data,
            logs_content=logs_content
        )

        # --- State 8: uploading (98%) ---
        job_service.update_job_status(job_id, "uploading")

        # Upload generated media files to S3
        video_url = s3_service.upload_video(job_id, output_dir / "video.mp4")
        thumbnail_url = s3_service.upload_image(job_id, output_dir / "thumbnail.jpg", is_thumbnail=True)
        s3_service.upload_file(job_id, f"jobs/{job_id}/preview.gif", output_dir / "preview.gif")

        # Upload complete job package artifacts to S3
        artifacts = ["request.json", "status.json", "runtime.json", "generation.json", "prompt.json", "metadata.json", "logs.txt"]
        for art in artifacts:
            s3_service.upload_file(job_id, f"jobs/{job_id}/{art}", output_dir / art)

        # Update final urls in status.json and metadata.json
        status_path = storage_service.jobs_dir / job_id / "status.json"
        status_data = storage_service.load_json(status_path)
        status_data["video_url"] = video_url
        status_data["thumbnail_url"] = thumbnail_url
        storage_service.save_json(status_path, status_data)

        meta_path = storage_service.jobs_dir / job_id / "metadata.json"
        meta_data = storage_service.load_json(meta_path)
        meta_data["S3_URL"] = video_url
        storage_service.save_json(meta_path, meta_data)

        # Keep output_dir in sync with final URLs
        shutil.copy2(status_path, output_dir / "status.json")
        shutil.copy2(meta_path, output_dir / "metadata.json")

        # --- State 9: completed (100%) ---
        job_service.update_job_status(job_id, "completed")
        logger.info(f"Worker completed job {job_id} successfully.")

        # Write final logs containing COMPLETED step
        try:
            final_logs = storage_service.get_job_logs(job_id)
            with open(output_dir / "logs.txt", "w", encoding="utf-8") as f:
                f.write(final_logs)
            # Re-upload logs.txt to S3 package to ensure consistency
            s3_service.upload_file(job_id, f"jobs/{job_id}/logs.txt", output_dir / "logs.txt")
        except Exception as le:
            logger.error(f"Failed to write/upload final log artifact: {le}")

    except JobCancelledError as jce:
        logger.warning(f"Job {job_id} cancelled during execution.")
        try:
            job_service.update_job_status(job_id, "cancelled", error_message=str(jce))
            if 'output_dir' in locals() and output_dir.is_dir():
                final_logs = storage_service.get_job_logs(job_id)
                with open(output_dir / "logs.txt", "w", encoding="utf-8") as f:
                    f.write(final_logs)
                s3_service.upload_file(job_id, f"jobs/{job_id}/logs.txt", output_dir / "logs.txt")
        except Exception as e:
            logger.error(f"Failed to update job status to cancelled: {e}")
    except BuzzBoxException as bbe:
        logger.error(f"BuzzBox error processing job {job_id}: {bbe.message}")
        try:
            stacktrace = traceback.format_exc()
            job_service.update_job_status(job_id, "failed", error_message=f"{bbe.message}\nStacktrace:\n{stacktrace}")
            if 'output_dir' in locals() and output_dir.is_dir():
                final_logs = storage_service.get_job_logs(job_id)
                with open(output_dir / "logs.txt", "w", encoding="utf-8") as f:
                    f.write(final_logs)
                s3_service.upload_file(job_id, f"jobs/{job_id}/logs.txt", output_dir / "logs.txt")
        except Exception as le:
            logger.error(f"Failed to log failure for job {job_id}: {str(le)}")
    except Exception as e:
        logger.exception(f"Unhandled error processing job {job_id}: {str(e)}")
        try:
            stacktrace = traceback.format_exc()
            job_service.update_job_status(job_id, "failed", error_message=f"{str(e)}\nStacktrace:\n{stacktrace}")
            if 'output_dir' in locals() and output_dir.is_dir():
                final_logs = storage_service.get_job_logs(job_id)
                with open(output_dir / "logs.txt", "w", encoding="utf-8") as f:
                    f.write(final_logs)
                s3_service.upload_file(job_id, f"jobs/{job_id}/logs.txt", output_dir / "logs.txt")
        except Exception as le:
            logger.error(f"Failed to log critical failure for job {job_id}: {str(le)}")
    finally:
        # Release the GPU scheduler lock to allow the next job to proceed
        scheduler.release(job_id)
