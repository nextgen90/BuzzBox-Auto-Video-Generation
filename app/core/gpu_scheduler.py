import threading
import time
from typing import List, Set, Dict, Any, Optional
from loguru import logger

class GPUScheduler:
    """
    GPUScheduler is a singleton coordinator that serializes GPU access.
    It manages the generation queue, tracks running jobs, and acts as a central
    registry for job cancellation flags.
    """
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super(GPUScheduler, cls).__new__(cls)
                cls._instance._initialized_singleton = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized_singleton", False):
            return
        
        self._gpu_lock = threading.Lock()
        self._queue: List[str] = []
        self._running: List[str] = []
        self._cancelled_jobs: Set[str] = set()
        self._state_lock = threading.Lock()
        
        self._initialized_singleton = True
        logger.info("GPUScheduler initialized.")

    def enqueue(self, job_id: str) -> None:
        """Appends a job ID to the execution queue if not already present."""
        with self._state_lock:
            if job_id not in self._queue and job_id not in self._running:
                self._queue.append(job_id)
                logger.info(f"Job '{job_id}' enqueued in GPUScheduler. Current queue: {len(self._queue)}")

    def dequeue(self, job_id: str) -> None:
        """Removes a job ID from the scheduler queue."""
        with self._state_lock:
            if job_id in self._queue:
                self._queue.remove(job_id)
                logger.info(f"Job '{job_id}' dequeued from GPUScheduler.")

    def acquire(self, job_id: str, timeout: float = 3600.0) -> bool:
        """
        Acquires the GPU lock for a job, blocking until available.
        Moves the job from the waiting queue to the running list.
        """
        self.enqueue(job_id)
        
        logger.info(f"Job '{job_id}' is waiting for GPU lock...")
        start_time = time.perf_counter()
        
        # Block until the GPU lock is acquired
        acquired = self._gpu_lock.acquire(timeout=timeout)
        if not acquired:
            logger.error(f"Job '{job_id}' timed out waiting for GPU lock after {timeout}s.")
            self.dequeue(job_id)
            return False
            
        with self._state_lock:
            if job_id in self._queue:
                self._queue.remove(job_id)
            if job_id not in self._running:
                self._running.append(job_id)
            logger.info(f"Job '{job_id}' successfully acquired GPU lock. Wait time: {time.perf_counter() - start_time:.2f}s")
            
        return True

    def release(self, job_id: str) -> None:
        """Releases the GPU lock and removes the job from the running list."""
        with self._state_lock:
            if job_id in self._running:
                self._running.remove(job_id)
            
            # Clean up cancellation state if present
            if job_id in self._cancelled_jobs:
                self._cancelled_jobs.remove(job_id)
                
        # Release the lock if held
        try:
            self._gpu_lock.release()
            logger.info(f"Job '{job_id}' released GPU lock.")
        except RuntimeError:
            # Raised if lock is not acquired by current thread, ignore or log
            logger.warning(f"Failed to release GPU lock for Job '{job_id}' - lock was not held.")

    def cancel(self, job_id: str) -> None:
        """Registers a job ID as cancelled, signaling the active engine to abort."""
        with self._state_lock:
            self._cancelled_jobs.add(job_id)
            logger.info(f"Job '{job_id}' has been registered as CANCELLED in GPUScheduler.")

    def is_cancelled(self, job_id: str) -> bool:
        """Checks if the given job ID has been registered as cancelled."""
        with self._state_lock:
            return job_id in self._cancelled_jobs

    def queue_length(self) -> int:
        """Returns the number of jobs waiting in the queue."""
        with self._state_lock:
            return len(self._queue)

    def running_jobs(self) -> List[str]:
        """Returns a copy of the list of currently running jobs."""
        with self._state_lock:
            return list(self._running)

    def available(self) -> bool:
        """Checks if the GPU resource is currently unlocked and available."""
        return not self._gpu_lock.locked()
