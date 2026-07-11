import os
import json
import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.core.exceptions import CharacterMissingError, BibleMissingError, IdentityMissingError
from app.services.storage_service import StorageService
from app.services.character_service import CharacterService
from app.services.prompt_service import PromptService
from app.services.runtime_service import RuntimeService
from app.services.job_service import JobService
from app.workers.generation_worker import process_generation_job

client = TestClient(app)

@pytest.fixture
def setup_temp_character():
    """Fixture to create a temporary test character in storage/characters/"""
    char_id = "test_sprint2_char"
    char_dir = settings.CHARACTERS_DIR / char_id
    
    # Create directories
    (char_dir / "identity").mkdir(parents=True, exist_ok=True)
    (char_dir / "poses").mkdir(parents=True, exist_ok=True)
    (char_dir / "voices").mkdir(parents=True, exist_ok=True)
    (char_dir / "bible").mkdir(parents=True, exist_ok=True)
    
    # Create metadata.json
    metadata = {
        "id": char_id,
        "display_name": "Test Sprint 2 Character",
        "version": "1.2",
        "default_voice": "test_voice",
        "default_style": "cinematic",
        "languages": ["en", "fr"],
        "identity_version": 2,
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


def test_storage_service() -> None:
    """Verifies that StorageService correctly initializes workspace and performs operations."""
    storage = StorageService()
    assert storage.base_dir.is_dir()
    assert storage.characters_dir.is_dir()
    assert storage.jobs_dir.is_dir()
    assert storage.prompts_dir.is_dir()
    
    # Test JSON write/read
    test_path = storage.base_dir / "test_write.json"
    data = {"test_key": "test_value"}
    storage.save_json(test_path, data)
    assert test_path.is_file()
    
    loaded = storage.load_json(test_path)
    assert loaded == data
    test_path.unlink()


def test_character_service_success(setup_temp_character) -> None:
    """Tests loading a valid character and parsing the markdown bible content."""
    char_id = setup_temp_character
    service = CharacterService()
    
    # Test list
    characters = service.list_characters()
    assert char_id in characters
    
    # Test load
    profile = service.load_character(char_id)
    assert profile.id == char_id
    assert profile.display_name == "Test Sprint 2 Character"
    assert profile.default_voice == "test_voice"
    assert profile.languages == ["en", "fr"]
    assert "character.md" in profile.bible_path
    assert "This is a test markdown character bible" in profile.bible_content
    assert len(profile.identity_images) == 1
    assert len(profile.poses) == 1
    assert len(profile.voices) == 1


def test_character_service_failures(setup_temp_character) -> None:
    """Tests that CharacterService validation raises correct exceptions on missing assets."""
    char_id = setup_temp_character
    char_dir = settings.CHARACTERS_DIR / char_id
    service = CharacterService()
    
    # Non-existent character
    with pytest.raises(CharacterMissingError):
        service.load_character("non_existent_character_uuid")
        
    # Missing bible
    bible_file = char_dir / "bible" / "character.md"
    bible_file.unlink()
    with pytest.raises(BibleMissingError):
        service.load_character(char_id)
        
    # Restore bible, delete identity image
    with open(bible_file, "w", encoding="utf-8") as f:
        f.write("# Restored")
    identity_file = char_dir / "identity" / "front.png"
    identity_file.unlink()
    with pytest.raises(IdentityMissingError):
        service.load_character(char_id)


def test_prompt_service(setup_temp_character) -> None:
    """Tests that PromptService merges character context and exports segmented outputs."""
    char_id = setup_temp_character
    char_service = CharacterService()
    profile = char_service.load_character(char_id)
    
    prompt_service = PromptService()
    options = {
        "duration": 45,
        "camera": "wide shot",
        "style": "anime",
        "negative_prompt": "high contrast"
    }
    script = "Action sequence begins now."
    
    prompt_data = prompt_service.build_prompt(script, profile, options)
    assert prompt_data["script"] == script
    assert prompt_data["character"] == char_id
    assert prompt_data["duration"] == 45
    assert prompt_data["camera"] == "wide shot"
    assert prompt_data["style"] == "anime"
    assert prompt_data["negative_prompt"] == "high contrast"
    assert profile.identity_images[0] in prompt_data["identity_reference"]
    
    # Test export
    job_id = "test_prompt_job_uuid"
    prompt_service.export_prompt_json(job_id, prompt_data)
    
    job_prompt_dir = settings.PROMPTS_DIR / f"job_{job_id}"
    assert job_prompt_dir.is_dir()
    assert (job_prompt_dir / "raw_prompt.json").is_file()
    assert (job_prompt_dir / "compiled_prompt.json").is_file()
    assert (job_prompt_dir / "negative_prompt.txt").is_file()
    
    # Check negative prompt content
    with open(job_prompt_dir / "negative_prompt.txt", "r", encoding="utf-8") as f:
        assert f.read() == "high contrast"
        
    # Cleanup
    shutil.rmtree(job_prompt_dir)


def test_job_service() -> None:
    """Tests JobService lifecycle, transitions, logging, status check, and list queries."""
    job_service = JobService()
    
    payload = {
        "request_id": "job-service-test-req",
        "character": "cyberpunk_nomad",
        "language": "en-US",
        "script": "Hello future.",
        "duration": 30,
        "aspect_ratio": "16:9",
        "resolution": "1080p",
        "fps": 30,
        "camera": "close up",
        "style": "cyberpunk",
        "negative_prompt": "cartoon",
        "metadata": {"priority": "high"}
    }
    
    # Create Job
    job_info = job_service.create_job(payload)
    job_id = job_info["job_id"]
    assert job_info["status"] == "queued"
    assert job_info["progress"] == 0
    
    job_dir = settings.JOBS_DIR / job_id
    assert job_dir.is_dir()
    assert (job_dir / "request.json").is_file()
    assert (job_dir / "status.json").is_file()
    assert (job_dir / "metadata.json").is_file()
    assert (job_dir / "timestamps.json").is_file()
    assert (job_dir / "logs.txt").is_file()
    
    # Get status
    status_data = job_service.get_job_status(job_id)
    assert status_data["job_id"] == job_id
    assert status_data["status"] == "queued"
    assert status_data["progress"] == 0
    assert status_data["estimated_remaining_seconds"] == 70  # (30 * 2) + 10 = 70
    
    # Update Status
    job_service.update_job_status(job_id, "loading_character")
    status_data = job_service.get_job_status(job_id)
    assert status_data["status"] == "loading_character"
    assert status_data["progress"] in {20, 25}
    assert status_data["estimated_remaining_seconds"] in {56, 52}  # 70 * (1 - 0.20) = 56, or 70 * (1 - 0.25) = 52.5 -> 52
    
    # Check List
    all_jobs = job_service.list_jobs()
    assert any(j["job_id"] == job_id for j in all_jobs)
    
    # Cleanup
    shutil.rmtree(job_dir)


def test_gpu_endpoints() -> None:
    """Verifies that /gpu and /runtime endpoints returns proper system diagnostics dictionary."""
    response = client.get("/gpu")
    assert response.status_code == 200
    data = response.json()
    assert "gpu" in data
    assert "cuda" in data
    assert "vram" in data
    assert "torch" in data
    assert "disk" in data
    assert "ram" in data
    assert "python" in data
    assert data["status"] in {"ready", "cpu_fallback"}
    
    # Test /runtime
    response_rt = client.get("/runtime")
    assert response_rt.status_code == 200
    assert response_rt.json() == data


def test_health_endpoint() -> None:
    """Verifies the expanded healthcheck endpoint returns detailed status parameters."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["application"] == "BuzzBox AI Engine"
    assert data["status"] in {"healthy", "degraded"}
    assert "gpu" in data
    assert "cuda" in data
    assert "torch" in data
    assert data["character_storage"] == "ready"
    assert data["jobs"] == "ready"


def test_generation_worker(setup_temp_character) -> None:
    """Tests the asynchronous generation worker end-to-end by calling it synchronously."""
    char_id = setup_temp_character
    job_service = JobService()
    
    payload = {
        "request_id": "worker-test-req",
        "character": char_id,
        "language": "en",
        "script": "A test script run inside the background worker.",
        "duration": 15,
        "aspect_ratio": "9:16",
        "resolution": "720p",
        "fps": 30,
        "camera": "medium-close-up",
        "style": "photorealistic",
        "negative_prompt": "deformed hands",
        "metadata": {"priority": "medium"}
    }
    
    # Create job in queued state
    job_info = job_service.create_job(payload)
    job_id = job_info["job_id"]
    
    # Process job synchronously
    process_generation_job(job_id)
    
    # Verify job completed successfully
    status_data = job_service.get_job_status(job_id)
    assert status_data["status"] == "completed"
    assert status_data["progress"] == 100
    assert status_data["estimated_remaining_seconds"] == 0
    assert status_data["error"] is None
    
    # Verify prompt files exist
    job_prompt_dir = settings.PROMPTS_DIR / f"job_{job_id}"
    assert job_prompt_dir.is_dir()
    assert (job_prompt_dir / "raw_prompt.json").is_file()
    assert (job_prompt_dir / "compiled_prompt.json").is_file()
    assert (job_prompt_dir / "negative_prompt.txt").is_file()
    
    # Verify logs file has multiple records and ends with COMPLETED
    job_dir = settings.JOBS_DIR / job_id
    with open(job_dir / "logs.txt", "r", encoding="utf-8") as f:
        logs_content = f.read()
        assert "[VALIDATING_REQUEST]" in logs_content
        assert "[LOADING_CHARACTER]" in logs_content
        assert "[LOADING_MODEL]" in logs_content
        assert "[BUILDING_PROMPT]" in logs_content
        assert "[RUNTIME_CHECK]" in logs_content
        assert "[WAITING_GPU]" in logs_content
        assert "[GENERATING]" in logs_content
        assert "[UPLOADING]" in logs_content
        assert "[COMPLETED]" in logs_content

    # Cleanup
    shutil.rmtree(job_dir)
    shutil.rmtree(job_prompt_dir)
