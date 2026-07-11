from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

def get_utc_timestamp() -> str:
    """
    Returns the current UTC time formatted as an ISO 8601 string.
    """
    return datetime.now(timezone.utc).isoformat()

def get_resolution_dimensions(resolution: str) -> Tuple[int, int]:
    """
    Translates standard resolution labels to width and height tuples.
    Returns (width, height). Defaults to 1920x1080 if label is unknown.
    """
    mapping: Dict[str, Tuple[int, int]] = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160),
        "1280x720": (1280, 720),
        "1920x1080": (1920, 1080),
        "3840x2160": (3840, 2160)
    }
    return mapping.get(resolution.lower().strip(), (1920, 1080))

def safe_delete_file(file_path: Path) -> bool:
    """
    Safely deletes a file if it exists. Returns True if deleted, otherwise False.
    """
    try:
        if file_path.is_file():
            file_path.unlink()
            return True
    except Exception:
        pass
    return False

def sanitize_for_json(data: any) -> any:
    """
    Recursively cleans data structures to ensure they are fully JSON serializable.
    Converts Exception objects (like ValueError) to their string representation.
    """
    import json
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, Exception):
        return str(data)
    else:
        try:
            json.dumps(data)
            return data
        except (TypeError, OverflowError):
            return str(data)
