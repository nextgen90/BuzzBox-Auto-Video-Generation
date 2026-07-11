import os
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)

def test_generate_video_success() -> None:
    """Verifies that a valid video generation request is accepted and queued."""
    payload = {
        "request_id": "test-req-001",
        "character": "Gothic Knight",
        "language": "en",
        "script": "A dark knight stands in the courtyard during a storm.",
        "duration": 30,
        "aspect_ratio": "16:9",
        "resolution": "1080p",
        "fps": 30,
        "background_music": True,
        "subtitles": False,
        "metadata": {"source": "unit-test"}
    }
    
    response = client.post("/generate-video", json=payload)
    assert response.status_code == 200
    
    json_data = response.json()
    assert json_data["success"] is True
    assert "job_id" in json_data
    assert json_data["status"] == "queued"
    assert json_data["message"] == "Video generation request accepted."
    assert "estimated_time" in json_data
    
    # Verify execution timing headers exist
    assert "X-Process-Time" in response.headers
    
    # Check that a JSON file was actually written to the storage directory
    job_id = json_data["job_id"]
    job_dir = settings.JOBS_DIR / job_id
    job_file = job_dir / "request.json"
    
    assert job_file.is_file() is True
    
    # Clean up test-generated job directory
    import shutil
    try:
        shutil.rmtree(job_dir)
    except Exception:
        pass
