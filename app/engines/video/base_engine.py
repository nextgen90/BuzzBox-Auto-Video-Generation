import abc
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
from app.engines.video.generation_result import GenerationResult

class BaseVideoEngine(abc.ABC):
    """
    BaseVideoEngine defines the strict abstract interface for all video generation engines
    in the BuzzBox platform. This design adheres to the Open/Closed Principle, allowing
    new video models (e.g., Kling, Veo, Hunyuan) to be plugged in seamlessly.
    """

    @abc.abstractmethod
    def initialize(self) -> None:
        """
        Initializes the pipeline, loads weights into VRAM/RAM, and runs optimizations.
        Must be called once before any generation requests.
        """
        pass

    @abc.abstractmethod
    def generate(
        self, 
        prompt: Dict[str, Any], 
        callback: Optional[Callable[[int, int, Any], None]] = None
    ) -> GenerationResult:
        """
        Executes video generation given compiled prompts and option overrides.
        
        Args:
            prompt: Merged prompt instruction containing script, character metadata, and visual cues.
            callback: Optional step callback to monitor progress and support execution cancellation.
            
        Returns:
            A standardized GenerationResult object.
        """
        pass

    @abc.abstractmethod
    def cleanup(self) -> None:
        """
        Unloads models, clears VRAM caches, and releases memory resources.
        """
        pass

    @abc.abstractmethod
    def health(self) -> Dict[str, Any]:
        """
        Returns engine health status, initialization state, and memory footprints.
        """
        pass

    @abc.abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """
        Returns supported parameters, configurations, and features of this engine.
        """
        pass

    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns the registered name of the engine (e.g., 'wan22').
        """
        pass

    @abc.abstractmethod
    def version(self) -> str:
        """
        Returns the version string of the engine.
        """
        pass

    @abc.abstractmethod
    def supported_resolutions(self) -> List[str]:
        """
        Returns the list of supported resolution strings (e.g., ['720p', '1080p']).
        """
        pass

    @abc.abstractmethod
    def supported_aspect_ratios(self) -> List[str]:
        """
        Returns the list of supported aspect ratio strings (e.g., ['16:9', '9:16']).
        """
        pass

    @abc.abstractmethod
    def supported_duration(self) -> int:
        """
        Returns the maximum supported video duration in seconds.
        """
        pass
