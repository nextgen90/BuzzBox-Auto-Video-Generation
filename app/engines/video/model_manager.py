import os
import threading
import time
from typing import Any, Dict, Optional, Type
from loguru import logger

from app.core.config import settings
from app.engines.video.base_engine import BaseVideoEngine
from app.engines.video.wan22_engine import Wan22Engine

class ModelManager:
    """
    ModelManager is a thread-safe singleton that manages the lifecycle of the
    active video generation engine. It loads the engine once, handles warmup,
    releases memory, monitors GPU VRAM usage, and provides automatic reloads on failure.
    """
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized_singleton = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized_singleton", False):
            return
        self._engine: Optional[BaseVideoEngine] = None
        self._engine_name: Optional[str] = None
        self._lock = threading.Lock()
        self._warmup_done = False
        self._initialized_singleton = True

    def load(self, engine_name: Optional[str] = None) -> None:
        """Loads and initializes the selected video generation engine."""
        with self._lock:
            target_engine = engine_name or settings.VIDEO_ENGINE
            
            # If already loaded and matching target, skip
            if self._engine is not None and self._engine_name == target_engine:
                logger.info(f"Engine '{target_engine}' is already loaded.")
                return

            logger.info(f"Loading video engine: {target_engine}")
            self._engine_name = target_engine

            # Map engine names to classes (Open/Closed Principle)
            engines_map: Dict[str, Type[BaseVideoEngine]] = {
                "wan22": Wan22Engine
                # Future engines can be registered here:
                # "hunyuan": HunyuanEngine,
                # "kling": KlingEngine,
            }

            if target_engine not in engines_map:
                raise ValueError(f"Unsupported video engine type: '{target_engine}'")

            # Instantiate and initialize
            engine_cls = engines_map[target_engine]
            self._engine = engine_cls()
            self._engine.initialize()
            logger.info(f"Engine '{target_engine}' loaded and initialized successfully.")

    def get_engine(self) -> BaseVideoEngine:
        """Retrieves the active video engine, loading it if necessary."""
        if self._engine is None:
            self.load()
        return self._engine

    def get_pipeline(self) -> Any:
        """
        Alias of get_engine for compatibility.
        Returns the BaseVideoEngine wrapper instance.
        """
        return self.get_engine()

    def is_loaded(self) -> bool:
        """Checks if the engine is initialized and ready for execution."""
        return self._engine is not None and getattr(self._engine, "is_initialized", False)

    def unload(self) -> None:
        """Unloads the current engine and triggers garbage collection to free VRAM/RAM."""
        with self._lock:
            if self._engine is not None:
                logger.info(f"Unloading engine '{self._engine_name}'")
                self._engine.cleanup()
                self._engine = None
                self._engine_name = None
                self._warmup_done = False
                
                # Force Python garbage collection
                import gc
                gc.collect()
                logger.info("Engine unloaded successfully.")

    def reload(self) -> None:
        """Re-initializes the engine, clearing out existing instances first."""
        logger.info("Triggering model reload...")
        current_name = self._engine_name
        self.unload()
        self.load(current_name)

    def warmup(self) -> None:
        """
        Runs a lightweight warmup execution through the pipeline.
        This forces CUDA kernel compilation and initialization to prevent
        first-request latencies for production users.
        """
        if not self.is_loaded():
            self.load()

        with self._lock:
            if self._warmup_done:
                logger.info("Engine has already completed warmup. Skipping.")
                return

            logger.info(f"Starting warmup inference pass for engine '{self._engine_name}'...")
            start_time = time.perf_counter()
            
            # Formulate a tiny warmup prompt
            warmup_prompt = {
                "script": "Warmup",
                "character": "warmup_char",
                "duration": 1,
                "fps": 15,
                "resolution": "720p",
                "aspect_ratio": "16:9",
                "seed": 42,
                "steps": 1,
                "guidance_scale": 1.0,
                "prompt_instruction": "Warmup inference pass"
            }

            try:
                # Run generate directly through engine (ignoring/discarding output files)
                res = self._engine.generate(warmup_prompt)
                
                # Cleanup temp files created by warmup
                for path in [res.video_path, res.thumbnail_path, res.preview_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                
                self._warmup_done = True
                logger.info(f"Warmup inference completed successfully in {time.perf_counter() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Warmup inference failed: {e}. Reloading engine.")
                # If warmup fails, reload the engine
                self._lock.release() # Release before calling reload to avoid deadlock
                try:
                    self.reload()
                finally:
                    self._lock.acquire() # Reacquire

    def memory_usage(self) -> Dict[str, Any]:
        """Queries current system and GPU VRAM memory levels."""
        stats = {
            "allocated": "0.0MB",
            "allocated_bytes": 0,
            "reserved": "0.0MB",
            "reserved_bytes": 0,
            "peak": "0.0MB",
            "peak_bytes": 0,
            "system_ram": "unknown"
        }
        
        # Check system memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            stats["system_ram"] = f"{(mem.total - mem.available) / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB"
        except Exception:
            pass

        # Check CUDA GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(0)
                res = torch.cuda.memory_reserved(0)
                peak = torch.cuda.max_memory_allocated(0)
                
                stats["allocated"] = f"{alloc / (1024**2):.1f}MB"
                stats["allocated_bytes"] = alloc
                stats["reserved"] = f"{res / (1024**2):.1f}MB"
                stats["reserved_bytes"] = res
                stats["peak"] = f"{peak / (1024**2):.1f}MB"
                stats["peak_bytes"] = peak
        except Exception as e:
            logger.warning(f"Failed to fetch VRAM metrics: {e}")

        return stats

    def get_loaded_model_info(self) -> Dict[str, Any]:
        """Returns loaded model details."""
        if not self.is_loaded():
            return {
                "loaded_model": "None",
                "engine": "None",
                "status": "unloaded",
                "memory": self.memory_usage()
            }
        
        return {
            "loaded_model": settings.MODEL_NAME,
            "engine": self._engine_name,
            "status": "loaded",
            "memory": self.memory_usage(),
            "health": self._engine.health() if self._engine else {}
        }
