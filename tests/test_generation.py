import os
import json
import shutil
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.engines.video.model_manager import ModelManager
from app.engines.video.wan22_engine import Wan22Engine
from app.core.gpu_scheduler import GPUScheduler
from app.services.s3_service import S3Service
from app.services.video_generation_service import VideoGenerationService
from app.services.job_service import JobService
from app.services.character_service import CharacterService
from app.workers.generation_worker import process_generation_job

client = TestClient(app)

@pytest.fixture
def setup_test_character():
    """Fixture to create a temporary test character in storage/characters/"""
    char_id = "test_sprint3_char"
    char_dir = settings.CHARACTERS_DIR / char_id
    
    # Create directories
    (char_dir / "identity").mkdir(parents=True, exist_ok=True)
    (char_dir / "poses").mkdir(parents=True, exist_ok=True)
    (char_dir / "voices").mkdir(parents=True, exist_ok=True)
    (char_dir / "bible").mkdir(parents=True, exist_ok=True)
    
    # Create metadata.json
    metadata = {
        "id": char_id,
        "display_name": "Test Sprint 3 Character",
        "version": "1.3",
        "default_voice": "test_voice",
        "default_style": "cinematic",
        "languages": ["en"],
        "identity_version": 1,
        "voice_version": 1,
        "status": "ready"
    }
    with open(char_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)
        
    # Create mock files
    with open(char_dir / "identity" / "front.png", "wb") as f:
        f.write(b"mock_image")
    with open(char_dir / "poses" / "standing.png", "wb") as f:
        f.write(b"mock_pose")
    with open(char_dir / "voices" / "sample.wav", "wb") as f:
        f.write(b"mock_voice")
    with open(char_dir / "bible" / "character.md", "w", encoding="utf-8") as f:
        f.write("# Character Bible\nThis is a test markdown character bible.")

    yield char_id

    # Clean up character folder after test
    if char_dir.is_dir():
        shutil.rmtree(char_dir)


def test_model_manager_singleton() -> None:
    """Verifies that ModelManager is a singleton and loads the engine."""
    manager1 = ModelManager()
    manager2 = ModelManager()
    assert manager1 is manager2

    # Verify properties when uninitialized/loaded
    assert not manager1.is_loaded()
    manager1.load()
    assert manager1.is_loaded()
    
    engine = manager1.get_engine()
    assert isinstance(engine, Wan22Engine)
    assert engine.name() == "wan22"
    assert engine.version() == "2.2.0"
    
    # Check health and capabilities
    health = engine.health()
    assert health["initialized"] is True
    
    caps = engine.capabilities()
    assert caps["supports_seed"] is True
    
    # Verify memory stats
    memory = manager1.memory_usage()
    assert "system_ram" in memory
    
    # Warmup pass
    manager1.warmup()
    assert manager1._warmup_done is True
    
    # Cleanup
    manager1.unload()
    assert not manager1.is_loaded()


def test_s3_service_mock_fallback() -> None:
    """Tests S3Service local mock fallback mechanism when AWS keys are absent."""
    # Force mock mode by setting empty credentials in settings if they are not already
    s3 = S3Service()
    assert s3.mock_mode is True

    # Test file upload
    temp_file = settings.OUTPUT_DIR / "test_s3_upload.txt"
    with open(temp_file, "w") as f:
        f.write("S3 mock content")
        
    job_id = "test_s3_job"
    url = s3.upload_file(job_id, f"jobs/{job_id}/test_s3_upload.txt", temp_file)
    assert url.startswith("/s3_mock/jobs/test_s3_job/")
    
    # Verify file is stored in mock directory
    assert (settings.STORAGE_DIR / "s3_mock" / f"jobs/{job_id}/test_s3_upload.txt").is_file()
    
    # Test video and image uploads
    v_url = s3.upload_video(job_id, temp_file)
    assert v_url.startswith("/s3_mock/videos/")
    
    t_url = s3.upload_image(job_id, temp_file, is_thumbnail=True)
    assert t_url.startswith("/s3_mock/thumbnails/")
    
    # Test delete
    assert s3.delete_object(f"jobs/{job_id}/test_s3_upload.txt") is True
    assert not (settings.STORAGE_DIR / "s3_mock" / f"jobs/{job_id}/test_s3_upload.txt").is_file()
    
    # Clean up
    temp_file.unlink()
    if (settings.STORAGE_DIR / "s3_mock" / "jobs" / job_id).is_dir():
        shutil.rmtree(settings.STORAGE_DIR / "s3_mock" / "jobs" / job_id)


def test_generation_service_prepare_and_profiles() -> None:
    """Verifies that VideoGenerationService correctly parses profiles and platforms."""
    service = VideoGenerationService()
    
    prompt_data = {
        "prompt_instruction": "Generate vertical tiktok clip of cyberpunk female nomad.",
        "character": "nomad",
        "language": "en",
        "duration": 10
    }
    
    # Case 1: TikTok platform and FAST profile
    request_params = {
        "platform": "tiktok",
        "profile": "FAST",
        "seed": 999
    }
    prepared = service.prepare(prompt_data, request_params)
    assert prepared["aspect_ratio"] == "9:16"
    assert prepared["resolution"] == "720p"
    assert prepared["steps"] == 10
    assert prepared["seed"] == 999
    assert prepared["fps"] == 30
    assert "Vertical video format" in prepared["prompt_instruction"]

    # Case 2: YouTube platform and ULTRA profile, with aspect ratio override
    request_params2 = {
        "platform": "youtube",
        "profile": "ULTRA",
        "aspect_ratio": "4:5"
    }
    prepared2 = service.prepare(prompt_data, request_params2)
    assert prepared2["aspect_ratio"] == "4:5" # overridden
    assert prepared2["resolution"] == "1080p"
    assert prepared2["steps"] == 50
    assert prepared2["seed"] > 0
    assert "Landscape cinematic video" in prepared2["prompt_instruction"]


def test_gpu_scheduler_locks_and_cancellation() -> None:
    """Verifies GPUScheduler enqueuing, locking, and cancellation flag checking."""
    scheduler = GPUScheduler()
    job_id = "test_sched_job_1"
    
    # Enqueue
    scheduler.enqueue(job_id)
    assert scheduler.queue_length() == 1
    
    # Acquire
    assert scheduler.available() is True
    acquired = scheduler.acquire(job_id)
    assert acquired is True
    assert scheduler.available() is False
    assert scheduler.queue_length() == 0
    assert job_id in scheduler.running_jobs()
    
    # Cancellation check
    assert scheduler.is_cancelled(job_id) is False
    scheduler.cancel(job_id)
    assert scheduler.is_cancelled(job_id) is True
    
    # Release
    scheduler.release(job_id)
    assert scheduler.available() is True
    assert job_id not in scheduler.running_jobs()


def test_generation_worker_lifecycle(setup_test_character) -> None:
    """Tests the worker executing the full Sprint 3 lifecycle and exporting outputs."""
    char_id = setup_test_character
    job_service = JobService()
    
    payload = {
        "request_id": "sprint3-worker-test",
        "character": char_id,
        "language": "en",
        "script": "A test script running in the updated worker.",
        "duration": 2,
        "platform": "tiktok",
        "profile": "FAST",
        "seed": 12345
    }
    
    # Create Job
    job_info = job_service.create_job(payload)
    job_id = job_info["job_id"]
    
    # Run worker synchronously
    process_generation_job(job_id)
    
    # Verify job status completed
    status_data = job_service.get_job_status(job_id)
    assert status_data["status"] == "completed"
    assert status_data["progress"] == 100
    assert status_data["video_url"].startswith("/s3_mock/videos/")
    assert status_data["thumbnail_url"].startswith("/s3_mock/thumbnails/")
    assert isinstance(status_data["generation_statistics"], dict)
    
    # Check outputs/job_id/ folder artifacts
    output_dir = settings.OUTPUT_DIR / job_id
    assert output_dir.is_dir()
    assert (output_dir / "video.mp4").is_file()
    assert (output_dir / "thumbnail.jpg").is_file()
    assert (output_dir / "preview.gif").is_file()
    assert (output_dir / "request.json").is_file()
    assert (output_dir / "status.json").is_file()
    assert (output_dir / "runtime.json").is_file()
    assert (output_dir / "generation.json").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "prompt.json").is_file()
    assert (output_dir / "logs.txt").is_file()
    
    # Check logs content
    with open(output_dir / "logs.txt", "r") as f:
        logs = f.read()
        assert "[VALIDATING]" in logs or "[VALIDATING_REQUEST]" in logs
        assert "[LOADING_MODEL]" in logs
        assert "[PREPARING_INPUTS]" in logs
        assert "[GENERATING]" in logs
        assert "[ENCODING]" in logs
        assert "[UPLOADING]" in logs
        assert "[COMPLETED]" in logs
        
    # Clean up
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    if (settings.JOBS_DIR / job_id).is_dir():
        shutil.rmtree(settings.JOBS_DIR / job_id)
    if (settings.PROMPTS_DIR / f"job_{job_id}").is_dir():
        shutil.rmtree(settings.PROMPTS_DIR / f"job_{job_id}")


def test_api_routes(setup_test_character) -> None:
    """Verifies all new and updated API routes endpoints."""
    char_id = setup_test_character
    
    # 1. Test GET /models
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "loaded_model" in data
    assert "engine" in data
    assert "status" in data
    
    # 2. Test GET /runtime is enriched
    response = client.get("/runtime")
    assert response.status_code == 200
    data = response.json()
    assert "gpu" in data
    assert "current_queue_length" in data
    assert "running_job_count" in data
    assert "model" in data
    
    # 3. Test POST /jobs/{job_id}/cancel on a queued job
    # We create the job directly via JobService so the worker is NOT dispatched
    job_service = JobService()
    payload = {
        "character": char_id,
        "language": "en-US",
        "script": "Manual queued cancellation script.",
        "duration": 5,
        "platform": "instagram",
        "profile": "STANDARD",
        "seed": 4567
    }
    job_info = job_service.create_job(payload)
    job_id = job_info["job_id"]

    # Now cancel the queued job via endpoint
    cancel_resp = client.post(f"/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["success"] is True

    # Verify status is cancelled
    status_resp = client.get(f"/jobs/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "cancelled"

    # 4. Test POST /generate-video triggers generation and completes
    payload_run = {
        "character": char_id,
        "language": "en-US",
        "script": "Synchronous worker execution script.",
        "duration": 2,
        "platform": "youtube",
        "profile": "FAST",
        "seed": 8888
    }
    # Under TestClient, this dispatches and runs the worker synchronously to completion
    response_run = client.post("/generate-video", json=payload_run)
    assert response_run.status_code == 200
    job_run_data = response_run.json()
    assert job_run_data["success"] is True
    job_id2 = job_run_data["job_id"]

    # Verify job status is completed
    status_run = client.get(f"/jobs/{job_id2}")
    assert status_run.status_code == 200
    assert status_run.json()["status"] == "completed"

    # 5. Test cancelling completed job raises 400 Bad Request
    cancel_err = client.post(f"/jobs/{job_id2}/cancel")
    assert cancel_err.status_code == 400

    # Clean up jobs
    for jid in [job_id, job_id2]:
        if (settings.JOBS_DIR / jid).is_dir():
            shutil.rmtree(settings.JOBS_DIR / jid)
        if (settings.OUTPUT_DIR / jid).is_dir():
            shutil.rmtree(settings.OUTPUT_DIR / jid)
        if (settings.PROMPTS_DIR / f"job_{jid}").is_dir():
            shutil.rmtree(settings.PROMPTS_DIR / f"job_{jid}")
