# BuzzBox AI Engine Backend - Sprint 3

BuzzBox AI Engine is an enterprise-grade, highly modular, and scalable backend foundation designed to power automated AI-driven video generation. This release extends the platform with the **Sprint 3: Real AI Video Generation Runtime (WAN 2.2 + GPU + S3)**, transitioning the engine from simulated progress to GPU-driven inference.

This runtime introduces a modular, pluggable engine architecture, serialized GPU scheduling, startup warming passes, platform-specific media optimization defaults, generation profiles, complete artifact package uploads to AWS S3, and real-time cancellation control.

---

## Features (Sprint 3)

- **Video Engine Abstraction**: Defines `BaseVideoEngine` and a uniform `GenerationResult` model. Swapping inference engines (e.g. from Wan 2.2 to Hunyuan, Kling, or Veo) is hot-swappable via configuration without any codebase refactoring.
- **Configurable Engine Selection**: Fully managed via environment variables (`VIDEO_ENGINE`, `MODEL_NAME`, `MODEL_CACHE`, `MODEL_PROVIDER`, `MODEL_PRECISION`, `DEVICE`).
- **Singleton Model Manager**: Handles lazy loading, automatic reloads on pipeline crash, hardware monitoring, and resources garbage collection.
- **Model Warmup**: Automatically runs startup inference passes on the selected engine during application startup to compile CUDA kernels and prevent first-request latency.
- **GPU Scheduler**: Serializes GPU execution utilizing a thread-safe scheduler, tracking waiting queues, and running tasks.
- **Job Cancellation**: Allows user requests to immediately halt rendering and release the GPU mid-inference using step-based callbacks.
- **Generation Profiles**: Exposes simplified quality levels (`FAST`, `STANDARD`, `QUALITY`, `ULTRA`) mapping to custom inference steps, guidance scales, schedulers, and precision parameters.
- **Platform Awareness**: Auto-optimizes aspect ratios, frame rates, and resolutions tailored for target platforms (`youtube`, `instagram`, `tiktok`, `facebook`, `future`).
- **AWS S3 Storage Packager**: Uploads the complete job directory (video, thumbnail, gif preview, request/status logs) under `jobs/{job_id}/` on S3, falling back to a local mounted HTTP folder for offline testing.

---

## Folder Structure

```text
backend/
    app/
        __init__.py
        main.py                 # Handles lifespans, warmup execution, request loggers, and mounts /s3_mock
        api/
            __init__.py
            routes.py           # API endpoints (includes POST /jobs/{id}/cancel and GET /models)
        core/
            __init__.py
            config.py           # Engine/model and AWS S3 properties loaded from .env
            exceptions.py       # Custom exception classes (adds JobCancelledError)
            gpu_scheduler.py    # Singleton scheduler locks and cancellation registries
            logging.py          # Loguru logger configurations
        engines/
            __init__.py
            video/
                __init__.py
                base_engine.py  # Abstract video generation engine class
                generation_result.py # Standardized inference result model
                wan22_engine.py # Wan 2.2 engine with Diffusers optimization and CPU fallback
                model_manager.py# Singleton engine loader and memory inspector
            voice/
            image/
        models/
            __init__.py
            character.py
            request.py          # VideoGenerationRequest with profile, platform, and optional parameters
            response.py
        services/
            __init__.py
            storage_service.py
            character_service.py
            prompt_service.py
            runtime_service.py
            job_service.py      # Expanded state machine, compatible metrics, and statistics loaders
            s3_service.py       # Boto3 client uploader with offline local storage mock
            video_generation_service.py # Resolves profiles/platforms and packages local outputs
            video_service.py
        utils/
            __init__.py
            helpers.py
        workers/
            __init__.py
            generation_worker.py# Orchestrator worker with GPU locking and step callbacks
    config/
    storage/
        characters/
        jobs/                   # Holds request.json, status.json, runtime.json, metadata.json, logs.txt, video.mp4, etc.
        prompts/
        s3_mock/                # Local mock S3 repository (videos/, thumbnails/, jobs/)
    logs/
    tests/
        test_main.py
        test_routes.py
        test_sprint2.py
        test_validation.py
        test_generation.py      # Sprint 3 comprehensive unit and integration test suite
    requirements.txt
    README.md
```

---

## Configuration Variables (`.env`)

Configure the video runtime by modifying backend environment parameters:

```bash
# Video Engine Settings
VIDEO_ENGINE="wan22"
MODEL_NAME="Wan-Video/Wan2.1-T2V-1.3B"
MODEL_CACHE="storage/cache"
MODEL_PROVIDER="huggingface"
MODEL_PRECISION="float16"
DEVICE="cuda"

# AWS Configuration (Empty values enable Local Mock S3 Mode)
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
S3_BUCKET=""
```

---

## Storage & S3 Layout

### 1. Job Output Package
Every completed job creates a packaged output folder locally under `outputs/{job_id}/` (and mirrored in `storage/jobs/{job_id}/`):
- `video.mp4`: Final synthesized video file.
- `thumbnail.jpg`: Covering JPEG frame image.
- `preview.gif`: Sub-sampled animated preview.
- `request.json`: Original incoming payload parameters.
- `status.json`: Granular status details.
- `prompt.json`: Compiled and merged text/visual contexts.
- `runtime.json`: Execution environment diagnostics (GPU, RAM, VRAM allocations).
- `generation.json`: Engine configuration and rendering timings.
- `metadata.json`: Search and display metadata properties (seed, resolution, scheduler, FPS).
- `logs.txt`: Sequential task events logging trace.

### 2. S3 Object Layout
The S3 client uploads items to the bucket using these object layouts:
- `videos/{job_id}/video.mp4`
- `thumbnails/{job_id}.jpg`
- `jobs/{job_id}/{artifact_name}` (uploads the entire output package)

---

## Job State Machine (Sprint 3)

Jobs transition sequentially through the following states:
1. `queued` (5%) - Ingestion accepted.
2. `validating` (15%) - Request parameter verification.
3. `loading_character` (25%) - Loads character profile and bibles.
4. `building_prompt` (40%) - Merges visual script contexts.
5. `loading_model` (60%) - Instantiates and caches model pipeline.
6. `preparing_inputs` (80%) - Allocates GPU lock, generates seeds, maps profiles/platforms.
7. `generating` (90%) - Running model diffusion loops.
8. `encoding` (95%) - Synthesizes frames to video and gif formats.
9. `uploading` (98%) - Uploading complete artifacts to S3.
10. `completed` (100%) - Completed successfully.
11. `failed` (100%) - Terminated due to errors.
12. `cancelled` (100%) - Aborted by user cancellation.

---

## Endpoint API Specifications (New & Enriched)

### 1. Inspect Loaded Models
- **Method & Route**: `GET /models`
- **Description**: Returns status on active video model engine and GPU memory.
- **Response Format**:
```json
{
  "loaded_model": "Wan-Video/Wan2.1-T2V-1.3B",
  "engine": "wan22",
  "status": "loaded",
  "memory": {
    "allocated": "3.5MB",
    "allocated_bytes": 3670016,
    "reserved": "12.0MB",
    "reserved_bytes": 12582912,
    "peak": "512.0MB",
    "peak_bytes": 536870912,
    "system_ram": "4.2GB / 16.0GB"
  },
  "health": {
    "name": "wan22",
    "version": "2.2.0",
    "initialized": true,
    "fallback_mode": false,
    "device": "cuda",
    "precision": "float16",
    "load_time_seconds": 2.45
  }
}
```

### 2. Cancel Active Job
- **Method & Route**: `POST /jobs/{job_id}/cancel`
- **Description**: Registers a job cancellation. Queued jobs are canceled instantly. Generating jobs are safely aborted on the next step.
- **Response Format**:
```json
{
  "success": true,
  "job_id": "9a3857db-5e16-43d1-bd8c-529a9ee11883",
  "message": "Cancellation request sent for job '9a3857db-5e16-43d1-bd8c-529a9ee11883'."
}
```

### 3. Get Enriched Job Status
- **Method & Route**: `GET /jobs/{job_id}`
- **Response Format**:
```json
{
  "job_id": "9a3857db-5e16-43d1-bd8c-529a9ee11883",
  "status": "completed",
  "progress": 100,
  "current_step": "Generation completed successfully",
  "estimated_remaining_seconds": 0,
  "estimated_time": 45,
  "video_url": "/s3_mock/videos/9a3857db-5e16-43d1-bd8c-529a9ee11883/video.mp4",
  "thumbnail_url": "/s3_mock/thumbnails/9a3857db-5e16-43d1-bd8c-529a9ee11883.jpg",
  "generation_statistics": {
    "method": "wan_pipeline_inference",
    "device": "cuda",
    "memory_allocated": "3.5MB",
    "memory_peak": "512.0MB",
    "steps_processed": 10
  },
  "error": null,
  "created_at": "2026-07-06T16:50:00Z",
  "updated_at": "2026-07-06T16:50:12Z"
}
```

---

## Verification & Testing

Verify that all Sprint 1, 2, and 3 capabilities pass without regression:

```bash
# Execute Sprint 3 tests
.\venv\Scripts\pytest tests/test_generation.py -v

# Execute complete suite
.\venv\Scripts\pytest tests/ -v
```
