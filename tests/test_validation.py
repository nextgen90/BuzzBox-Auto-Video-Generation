from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_validation_missing_fields() -> None:
    """Verifies that missing required fields trigger validation failure (422)."""
    # Empty payload
    response = client.post("/generate-video", json={})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error_code"] == "VALIDATION_ERROR"
    assert "details" in json_data

def test_validation_invalid_aspect_ratio() -> None:
    """Verifies that unsupported aspect ratio raises a validation error."""
    payload = {
        "character": "Robot",
        "language": "en",
        "script": "Self-destruction in 3... 2... 1...",
        "duration": 10,
        "aspect_ratio": "21:9",  # Unsupported aspect ratio
        "resolution": "1080p",
        "fps": 30
    }
    response = client.post("/generate-video", json=payload)
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert any("Aspect ratio" in err.get("msg", "") for err in json_data["details"])

def test_validation_invalid_resolution() -> None:
    """Verifies that unsupported resolutions raise a validation error."""
    payload = {
        "character": "Robot",
        "language": "en",
        "script": "Testing resolutions.",
        "duration": 10,
        "aspect_ratio": "16:9",
        "resolution": "8k",  # Unsupported resolution
        "fps": 30
    }
    response = client.post("/generate-video", json=payload)
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert any("Resolution" in err.get("msg", "") for err in json_data["details"])

def test_validation_negative_duration() -> None:
    """Verifies that negative duration fails validator constraints."""
    payload = {
        "character": "Robot",
        "language": "en",
        "script": "Negative duration test.",
        "duration": -5,  # Must be gt=0
        "aspect_ratio": "16:9",
        "resolution": "1080p",
        "fps": 30
    }
    response = client.post("/generate-video", json=payload)
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert any("greater than 0" in err.get("msg", "") for err in json_data["details"])

def test_validation_excessive_fps() -> None:
    """Verifies that extreme frame rates fail validation bounds."""
    payload = {
        "character": "Robot",
        "language": "en",
        "script": "Extreme frame rate test.",
        "duration": 10,
        "aspect_ratio": "16:9",
        "resolution": "1080p",
        "fps": 240  # Must be le=120
    }
    response = client.post("/generate-video", json=payload)
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert any("less than or equal to 120" in err.get("msg", "") for err in json_data["details"])
