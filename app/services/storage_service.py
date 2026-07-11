import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger
from app.core.config import settings
from app.core.exceptions import StorageError

class StorageService:
    """
    StorageService manages all dynamic file operations, ensuring directories
    and sub-directories exist, validating read/write constraints, and providing
    methods for thread-safe access to job assets, character profiles, and logs.
    """
    def __init__(self) -> None:
        self.base_dir = settings.STORAGE_DIR
        self.characters_dir = settings.CHARACTERS_DIR
        self.jobs_dir = settings.JOBS_DIR
        self.prompts_dir = settings.PROMPTS_DIR
        self.metadata_dir = settings.METADATA_DIR
        self.cache_dir = settings.CACHE_DIR
        self.exports_dir = settings.EXPORTS_DIR
        self.renders_dir = settings.RENDERS_DIR

        # Initialize folders
        self.initialize_directories()

    def initialize_directories(self) -> None:
        """Creates the complete workspace directory hierarchy if missing."""
        directories = [
            self.base_dir,
            self.characters_dir,
            self.jobs_dir,
            self.prompts_dir,
            self.metadata_dir,
            self.cache_dir,
            self.exports_dir,
            self.renders_dir
        ]
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create storage directory {directory}: {str(e)}")
                raise StorageError(f"Failed to initialize storage directory: {directory}. Detail: {str(e)}")

    def create_job_directory(self, job_id: str) -> Path:
        """Prepares a dedicated workspace folder for an active job."""
        job_path = self.jobs_dir / job_id
        try:
            job_path.mkdir(parents=True, exist_ok=True)
            return job_path
        except Exception as e:
            logger.error(f"Failed to create job directory for {job_id}: {str(e)}")
            raise StorageError(f"Failed to create folder structure for job {job_id}: {str(e)}")

    def save_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Writes dictionary payload as formatted JSON to the specified path."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed writing JSON data to {path}: {str(e)}")
            raise StorageError(f"File system write failure at {path.name}. Detail: {str(e)}")

    def load_json(self, path: Path) -> Dict[str, Any]:
        """Reads and parses JSON data from the specified path."""
        if not path.is_file():
            raise StorageError(f"Requested storage file does not exist: {path.name}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed reading JSON data from {path}: {str(e)}")
            raise StorageError(f"File system read failure at {path.name}. Detail: {str(e)}")

    def append_job_log(self, job_id: str, status: str, message: str) -> None:
        """Appends a progress log entry with timestamp and active status."""
        job_dir = self.create_job_directory(job_id)
        log_file = job_dir / "logs.txt"
        now_str = datetime.now(timezone.utc).isoformat()
        log_entry = f"[{now_str}] [{status.upper()}] {message}\n"
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
            logger.debug(f"Job {job_id} Log: {message}")
        except Exception as e:
            logger.error(f"Failed appending log for job {job_id}: {str(e)}")
            raise StorageError(f"Failed appending log entry for job {job_id}: {str(e)}")

    def get_job_logs(self, job_id: str) -> str:
        """Reads the full job execution log string."""
        log_file = self.jobs_dir / job_id / "logs.txt"
        if not log_file.is_file():
            return ""
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed reading logs for job {job_id}: {str(e)}")
            raise StorageError(f"Failed to read logs for job {job_id}: {str(e)}")

    def list_jobs(self) -> List[str]:
        """Lists all active and completed job UUID strings discovered in storage."""
        try:
            return [d.name for d in self.jobs_dir.iterdir() if d.is_dir()]
        except Exception as e:
            logger.error(f"Failed listing jobs directory: {str(e)}")
            raise StorageError(f"Failed querying job logs from storage. Detail: {str(e)}")
