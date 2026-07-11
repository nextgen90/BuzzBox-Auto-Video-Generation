import sys
import shutil
import subprocess
from typing import Any, Dict
from loguru import logger
from app.core.config import settings

class RuntimeService:
    """
    RuntimeService inspects system resources, CPU/GPU hardware capabilities,
    CUDA availability, VRAM size, and the status/versions of core AI dependencies.
    """
    
    def is_cuda_available(self) -> bool:
        """Checks if PyTorch is installed and CUDA is available for GPU operations."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def torch_version(self) -> str:
        """Returns current PyTorch installation version, or 'not_installed'."""
        try:
            import torch
            return torch.__version__
        except ImportError:
            return "not_installed"

    def get_library_version(self, module_name: str) -> str:
        """Checks if a python library is installed and returns its version."""
        try:
            mod = __import__(module_name)
            if hasattr(mod, "__version__"):
                return mod.__version__
            return "installed"
        except ImportError:
            return "not_installed"

    def system_memory(self) -> Dict[str, Any]:
        """Retrieves system RAM capacity (Total)."""
        total_bytes = 0
        
        # Try importing psutil if installed
        try:
            import psutil
            total_bytes = psutil.virtual_memory().total
        except ImportError:
            pass

        if total_bytes == 0:
            # OS specific fallbacks
            if sys.platform.startswith("win"):
                try:
                    out = subprocess.check_output(
                        "wmic computersystem get TotalPhysicalMemory", 
                        shell=True,
                        stderr=subprocess.DEVNULL
                    ).decode()
                    for line in out.splitlines():
                        line = line.strip()
                        if line and line.isdigit():
                            total_bytes = int(line)
                            break
                except Exception:
                    pass
            elif sys.platform.startswith("linux"):
                try:
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                parts = line.split()
                                if len(parts) >= 2:
                                    total_bytes = int(parts[1]) * 1024
                                    break
                except Exception:
                    pass

        # If all fail, mock a standard 16GB
        if total_bytes == 0:
            total_bytes = 16 * (1024**3)

        total_gb = total_bytes / (1024**3)
        return {
            "total": f"{total_gb:.1f}GB",
            "total_bytes": total_bytes
        }

    def gpu_information(self) -> Dict[str, Any]:
        """Queries GPU model name, CUDA driver version, and available VRAM."""
        cuda_avail = self.is_cuda_available()
        gpu_name = "None"
        vram_total = "0GB"
        driver_version = "unknown"

        # Check driver version via nvidia-smi
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            if out:
                driver_version = out.split("\n")[0]
        except Exception:
            try:
                out = subprocess.check_output("nvidia-smi", shell=True, stderr=subprocess.DEVNULL).decode()
                for line in out.splitlines():
                    if "Driver Version:" in line:
                        parts = line.split("Driver Version:")
                        driver_version = parts[1].strip().split()[0]
                        break
            except Exception:
                pass

        if cuda_avail:
            try:
                import torch
                gpu_name = torch.cuda.get_device_name(0)
                total_mem = torch.cuda.get_device_properties(0).total_memory
                vram_total = f"{total_mem / (1024**3):.1f}GB"
            except Exception as e:
                logger.warning(f"Error checking CUDA details via torch: {str(e)}")
        elif driver_version != "unknown":
            try:
                gpu_out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                if gpu_out:
                    parts = gpu_out.split(",")
                    gpu_name = parts[0].strip()
                    mem_mb = int(parts[1].strip())
                    vram_total = f"{mem_mb / 1024:.1f}GB"
            except Exception:
                pass

        return {
            "device": gpu_name,
            "vram": vram_total,
            "driver": driver_version
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Compiles a complete hardware and AI library dependency checklist report."""
        gpu_info = self.gpu_information()
        sys_mem = self.system_memory()
        
        # Calculate free disk space in GB
        try:
            total, used, free = shutil.disk_usage(settings.STORAGE_DIR)
            disk_free = f"{free / (1024**3):.1f}GB"
        except Exception:
            disk_free = "unknown"

        return {
            "gpu": gpu_info["device"],
            "cuda": self.is_cuda_available(),
            "vram": gpu_info["vram"],
            "torch": self.torch_version(),
            "diffusers": self.get_library_version("diffusers"),
            "transformers": self.get_library_version("transformers"),
            "accelerate": self.get_library_version("accelerate"),
            "disk": disk_free,
            "ram": sys_mem["total"],
            "python": sys.version.split()[0],
            "status": "ready" if self.is_cuda_available() else "cpu_fallback"
        }
