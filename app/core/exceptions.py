from fastapi import status

class BuzzBoxException(Exception):
    """Base exception for all BuzzBox errors."""
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "INTERNAL_SERVER_ERROR"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str = None, status_code: int = None, error_code: str = None) -> None:
        super().__init__(message or self.message)
        if message:
            self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_code:
            self.error_code = error_code

class CharacterMissingError(BuzzBoxException):
    status_code: int = status.HTTP_404_NOT_FOUND
    error_code: str = "CHARACTER_MISSING"
    message: str = "Character folder not found in storage."

class BibleMissingError(BuzzBoxException):
    status_code: int = status.HTTP_400_BAD_REQUEST
    error_code: str = "BIBLE_MISSING"
    message: str = "Required character bible file is missing."

class IdentityMissingError(BuzzBoxException):
    status_code: int = status.HTTP_400_BAD_REQUEST
    error_code: str = "IDENTITY_MISSING"
    message: str = "Required character identity images are missing."

class StorageError(BuzzBoxException):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "STORAGE_FAILURE"
    message: str = "Failed to perform file storage operation."

class GPUUnavailableError(BuzzBoxException):
    status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code: str = "GPU_UNAVAILABLE"
    message: str = "GPU runtime verification failed or hardware is unavailable."

class JobNotFoundError(BuzzBoxException):
    status_code: int = status.HTTP_404_NOT_FOUND
    error_code: str = "INVALID_JOB"
    message: str = "Requested job identifier not found."

class InvalidJobError(BuzzBoxException):
    status_code: int = status.HTTP_400_BAD_REQUEST
    error_code: str = "INVALID_JOB"
    message: str = "Job request parameters or state transitions are invalid."

class JobCancelledError(BuzzBoxException):
    status_code: int = 499
    error_code: str = "JOB_CANCELLED"
    message: str = "The job was cancelled by the user."
