import os
import shutil
import time
from pathlib import Path
from typing import Optional
from loguru import logger
import boto3
from botocore.exceptions import ClientError

from app.core.config import settings

class S3Service:
    """
    S3Service handles uploading generated videos, thumbnails, and job packages
    to AWS S3. If AWS credentials are not configured in settings, it falls back
    to a local mock mode which stores files locally under storage/s3_mock/ and
    returns valid local HTTP URLs.
    """
    def __init__(self) -> None:
        self.bucket_name = settings.S3_BUCKET
        self.region = settings.AWS_REGION
        self.mock_mode = True
        self.s3_client = None

        # Check if AWS credentials are set
        has_credentials = (
            settings.AWS_ACCESS_KEY_ID 
            and settings.AWS_SECRET_ACCESS_KEY 
            and self.bucket_name
        )

        if has_credentials:
            try:
                logger.info("Initializing S3 Client using configured environment variables.")
                self.s3_client = boto3.client(
                    "s3",
                    region_name=self.region,
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
                )
                # Verify bucket exists/accessible
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                self.mock_mode = False
                logger.info(f"S3 client initialized successfully. Bucket: {self.bucket_name}")
            except Exception as e:
                logger.warning(f"Failed to verify S3 bucket access ({e}). Falling back to local mock mode.")
                self.mock_mode = True
        else:
            logger.info("AWS S3 credentials not fully configured. Running in Local Mock S3 Mode.")

        if self.mock_mode:
            self.mock_dir = settings.STORAGE_DIR / "s3_mock"
            self.mock_dir.mkdir(parents=True, exist_ok=True)
            (self.mock_dir / "videos").mkdir(exist_ok=True)
            (self.mock_dir / "thumbnails").mkdir(exist_ok=True)
            (self.mock_dir / "metadata").mkdir(exist_ok=True)
            (self.mock_dir / "jobs").mkdir(exist_ok=True)

    def upload_video(self, job_id: str, file_path: Path) -> str:
        """
        Uploads video file. Returns the S3 URL or presigned URL (or local HTTP URL in mock mode).
        S3 target key: videos/{job_id}/video.mp4
        """
        s3_key = f"videos/{job_id}/video.mp4"
        return self.upload_file(job_id, s3_key, file_path)

    def upload_image(self, job_id: str, file_path: Path, is_thumbnail: bool = True) -> str:
        """
        Uploads image file. Returns the S3 URL (or local HTTP URL in mock mode).
        S3 target key: thumbnails/{job_id}.jpg
        """
        s3_key = f"thumbnails/{job_id}.jpg" if is_thumbnail else f"images/{job_id}.jpg"
        return self.upload_file(job_id, s3_key, file_path)

    def upload_file(self, job_id: str, s3_key: str, file_path: Path) -> str:
        """Uploads any file to S3 or stores it in the local mock directory."""
        if not file_path.is_file():
            raise FileNotFoundError(f"File to upload not found: {file_path}")

        if self.mock_mode:
            # Local mock upload
            target_path = self.mock_dir / s3_key
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target_path)
            
            # Return a local HTTP url served by FastAPI static mount
            url = f"/s3_mock/{s3_key}"
            logger.info(f"[Mock S3] Uploaded {file_path.name} to {s3_key}. URL: {url}")
            return url

        # Real S3 upload
        try:
            extra_args = {}
            # Auto detect content type
            if file_path.suffix.lower() == ".mp4":
                extra_args["ContentType"] = "video/mp4"
            elif file_path.suffix.lower() in {".jpg", ".jpeg"}:
                extra_args["ContentType"] = "image/jpeg"
            elif file_path.suffix.lower() == ".gif":
                extra_args["ContentType"] = "image/gif"
            elif file_path.suffix.lower() == ".json":
                extra_args["ContentType"] = "application/json"
            elif file_path.suffix.lower() == ".txt":
                extra_args["ContentType"] = "text/plain"

            logger.info(f"Uploading {file_path} to S3 bucket {self.bucket_name} as {s3_key}")
            self.s3_client.upload_file(
                Filename=str(file_path),
                Bucket=self.bucket_name,
                Key=s3_key,
                ExtraArgs=extra_args
            )
            
            # Generate a presigned URL or construct a public URL
            return self.generate_presigned_url(s3_key)
        except ClientError as ce:
            logger.error(f"S3 Upload failed for key {s3_key}: {ce}")
            raise ce

    def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """Generates a presigned URL to share/download the private S3 object."""
        if self.mock_mode:
            return f"/s3_mock/{s3_key}"

        try:
            response = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expiration
            )
            return response
        except ClientError as ce:
            logger.error(f"Failed to generate presigned URL for {s3_key}: {ce}")
            # Fallback to public URL construct
            return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"

    def delete_object(self, s3_key: str) -> bool:
        """Deletes an object from S3 or local mock folder."""
        if self.mock_mode:
            target_path = self.mock_dir / s3_key
            if target_path.is_file():
                try:
                    target_path.unlink()
                    logger.info(f"[Mock S3] Deleted object {s3_key}")
                    return True
                except Exception:
                    return False
            return False

        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted S3 object {s3_key}")
            return True
        except ClientError as ce:
            logger.error(f"Failed to delete S3 object {s3_key}: {ce}")
            return False

    def cleanup_old_jobs(self, days: int = 7) -> None:
        """Cleans up S3 files (or mock files) older than specified days."""
        cutoff_time = time.time() - (days * 86400)
        logger.info(f"Cleaning up job files older than {days} days...")

        if self.mock_mode:
            # Clean local mock folders
            for root_dir in [self.mock_dir / "videos", self.mock_dir / "thumbnails", self.mock_dir / "jobs"]:
                if not root_dir.exists():
                    continue
                for item in root_dir.iterdir():
                    try:
                        if item.is_file() and item.stat().st_mtime < cutoff_time:
                            item.unlink()
                        elif item.is_dir() and item.stat().st_mtime < cutoff_time:
                            shutil.rmtree(item)
                    except Exception as e:
                        logger.warning(f"Failed to clean mock S3 item {item}: {e}")
            return

        # Real S3 cleanup
        try:
            # We would list objects and delete those that match the time cutoff
            # For safety and pagination, this requires list_objects_v2 loops
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name):
                if 'Contents' not in page:
                    continue
                for obj in page['Contents']:
                    last_modified = obj['LastModified'].timestamp()
                    if last_modified < cutoff_time:
                        self.delete_object(obj['Key'])
        except Exception as e:
            logger.error(f"Failed during S3 old jobs cleanup: {e}")
