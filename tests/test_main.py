from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root() -> None:
    """Verifies that root URL returns proper branding indicators."""
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["application"] == "BuzzBox AI Engine"
    assert json_data["status"] == "running"

def test_read_health() -> None:
    """Verifies the healthcheck endpoint outputs correct metrics structure."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["application"] == "BuzzBox AI Engine"
    assert json_data["status"] in {"healthy", "degraded"}
    assert "gpu" in json_data
    assert "cuda" in json_data
    assert "torch" in json_data
    assert json_data["character_storage"] == "ready"
    assert json_data["jobs"] == "ready"
    assert json_data["version"] == "1.0.0"
