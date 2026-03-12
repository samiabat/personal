import os
import uuid
import json
import asyncio
import shutil
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="AI Video Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = "jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

# In-memory job tracking
jobs = {}

class Scene(BaseModel):
    voiceover: str
    prompt: str

class VideoRequest(BaseModel):
    scenes: list[Scene]
    speech_provider: str = "google"  # "google" or "openai"
    speech_model: str = "gemini-2.5-pro-preview-tts"
    speech_voice: str = "Charon"
    image_provider: str = "gemini"  # "gemini", "openai", or "togetherai"
    image_model: str = "gemini-3.1-flash-image-preview"
    aspect_ratio: str = "16:9"
    image_size: str = "512"
    openai_image_size: str = "1024x1024"
    togetherai_width: int = 1024
    togetherai_height: int = 576
    resolution: str = "1080p"
    enable_ken_burns: bool = False
    enable_zoom: bool = False
    enable_shake: bool = False

class TestAudioRequest(BaseModel):
    text: str
    speech_provider: str = "google"
    speech_model: str = "gemini-2.5-pro-preview-tts"
    speech_voice: str = "Charon"

class TestImageRequest(BaseModel):
    prompt: str
    image_provider: str = "gemini"
    image_model: str = "gemini-3.1-flash-image-preview"
    aspect_ratio: str = "16:9"
    image_size: str = "512"
    openai_image_size: str = "1024x1024"
    togetherai_width: int = 1024
    togetherai_height: int = 576


def _generate_image_for_provider(provider: str, prompt: str, image_path: str, **kwargs):
    """Route image generation to the correct provider."""
    if provider == "openai":
        from image_services import generate_image_openai
        return generate_image_openai(
            prompt, image_path,
            model=kwargs.get("image_model", "gpt-image-1"),
            size=kwargs.get("openai_image_size", "1024x1024"),
        )
    elif provider == "togetherai":
        from image_services import generate_image_togetherai
        return generate_image_togetherai(
            prompt, image_path,
            model=kwargs.get("image_model", "black-forest-labs/FLUX.1-schnell"),
            width=kwargs.get("togetherai_width", 1024),
            height=kwargs.get("togetherai_height", 576),
        )
    else:  # gemini
        from gemini_services import generate_image
        return generate_image(
            prompt, image_path,
            model=kwargs.get("image_model", "gemini-3.1-flash-image-preview"),
            aspect_ratio=kwargs.get("aspect_ratio", "16:9"),
            image_size=kwargs.get("image_size", "512"),
        )


def run_video_generation(job_id: str, request: VideoRequest):
    """Background task to generate video."""
    job = jobs[job_id]
    job_dir = f"{JOBS_DIR}/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    total_scenes = len(request.scenes)

    try:
        for index, scene in enumerate(request.scenes):
            # Update progress
            progress = int((index / total_scenes) * 90)  # 0-90% for scene generation
            job["progress"] = progress
            job["status"] = "processing"
            job["message"] = f"Generating scene {index + 1}/{total_scenes}..."

            audio_path = f"{job_dir}/scene_{index}_audio.wav"
            image_path = f"{job_dir}/scene_{index}_image.jpg"

            # Generate audio (skip if already exists - resume support)
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                job["message"] = f"Scene {index + 1}/{total_scenes}: Audio already exists, skipping..."
            else:
                job["message"] = f"Scene {index + 1}/{total_scenes}: Generating audio..."
                if request.speech_provider == "openai":
                    from openai_services import generate_audio_openai
                    generate_audio_openai(scene.voiceover, audio_path, model=request.speech_model, voice=request.speech_voice)
                else:
                    from gemini_services import generate_audio
                    generate_audio(scene.voiceover, audio_path, model=request.speech_model, voice=request.speech_voice)

            # Generate image (skip if already exists - resume support)
            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                job["message"] = f"Scene {index + 1}/{total_scenes}: Image already exists, skipping..."
            else:
                job["message"] = f"Scene {index + 1}/{total_scenes}: Generating image..."
                _generate_image_for_provider(
                    request.image_provider, scene.prompt, image_path,
                    image_model=request.image_model,
                    aspect_ratio=request.aspect_ratio,
                    image_size=request.image_size,
                    openai_image_size=request.openai_image_size,
                    togetherai_width=request.togetherai_width,
                    togetherai_height=request.togetherai_height,
                )

            job["progress"] = int(((index + 1) / total_scenes) * 90)

        # Assemble video
        job["progress"] = 92
        job["message"] = "Assembling final video..."
        from video_editor import assemble_final_video
        output_path = assemble_final_video(
            total_scenes, job_dir, "output.mp4",
            resolution=request.resolution,
            enable_ken_burns=request.enable_ken_burns,
            enable_zoom=request.enable_zoom,
            enable_shake=request.enable_shake,
        )

        # Clean up temp files
        job["progress"] = 98
        job["message"] = "Cleaning up temporary files..."
        for index in range(total_scenes):
            audio_path = f"{job_dir}/scene_{index}_audio.wav"
            image_path = f"{job_dir}/scene_{index}_image.jpg"
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(image_path):
                os.remove(image_path)

        job["progress"] = 100
        job["status"] = "completed"
        job["message"] = "Video generation completed!"
        job["output_path"] = output_path

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["progress"] = job.get("progress", 0)


@app.post("/api/generate-video")
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued...",
        "output_path": None,
        "request": request.model_dump(),
    }
    background_tasks.add_task(run_video_generation, job_id, request)
    return {"job_id": job_id}


@app.post("/api/retry/{job_id}")
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    """Retry a failed job, resuming from where it left off (skipping existing assets)."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "failed":
        raise HTTPException(status_code=400, detail="Only failed jobs can be retried")

    # Rebuild the request from stored data
    request = VideoRequest(**job["request"])

    # Reset job status
    job["status"] = "queued"
    job["progress"] = 0
    job["message"] = "Retrying job..."
    job["output_path"] = None

    background_tasks.add_task(run_video_generation, job_id, request)
    return {"job_id": job_id}


@app.get("/api/progress/{job_id}")
async def get_progress(job_id: str, request: Request):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            job = jobs.get(job_id)
            if not job:
                break
            yield {
                "event": "progress",
                "data": json.dumps({
                    "status": job["status"],
                    "progress": job["progress"],
                    "message": job["message"],
                })
            }
            if job["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed" or not job.get("output_path"):
        raise HTTPException(status_code=400, detail="Video not ready")
    return FileResponse(
        job["output_path"],
        media_type="video/mp4",
        filename="generated_video.mp4"
    )


@app.post("/api/test-audio")
async def test_audio(request: TestAudioRequest):
    test_id = str(uuid.uuid4())
    test_dir = f"{JOBS_DIR}/test_{test_id}"
    os.makedirs(test_dir, exist_ok=True)
    audio_path = f"{test_dir}/test_audio.wav"

    try:
        if request.speech_provider == "openai":
            from openai_services import generate_audio_openai
            success = generate_audio_openai(request.text, audio_path, model=request.speech_model, voice=request.speech_voice)
        else:
            from gemini_services import generate_audio
            success = generate_audio(request.text, audio_path, model=request.speech_model, voice=request.speech_voice)

        if success:
            return FileResponse(audio_path, media_type="audio/wav", filename="test_audio.wav")
        raise HTTPException(status_code=500, detail="Audio generation failed")
    except Exception as e:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/test-image")
async def test_image(request: TestImageRequest):
    test_id = str(uuid.uuid4())
    test_dir = f"{JOBS_DIR}/test_{test_id}"
    os.makedirs(test_dir, exist_ok=True)
    image_path = f"{test_dir}/test_image.jpg"

    try:
        success = _generate_image_for_provider(
            request.image_provider, request.prompt, image_path,
            image_model=request.image_model,
            aspect_ratio=request.aspect_ratio,
            image_size=request.image_size,
            openai_image_size=request.openai_image_size,
            togetherai_width=request.togetherai_width,
            togetherai_height=request.togetherai_height,
        )

        if success:
            return FileResponse(image_path, media_type="image/jpeg", filename="test_image.jpg")
        raise HTTPException(status_code=500, detail="Image generation failed")
    except Exception as e:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models():
    return {
        "speech_providers": [
            {"value": "google", "label": "Google Gemini"},
            {"value": "openai", "label": "OpenAI"},
        ],
        "speech_models": {
            "google": [
                {"value": "gemini-2.5-pro-preview-tts", "label": "Gemini 2.5 Pro TTS"},
                {"value": "gemini-2.5-flash-preview-tts", "label": "Gemini 2.5 Flash TTS"},
            ],
            "openai": [
                {"value": "tts-1", "label": "TTS-1"},
                {"value": "tts-1-hd", "label": "TTS-1 HD"},
                {"value": "gpt-4o-mini-tts", "label": "GPT-4o Mini TTS"},
            ],
        },
        "speech_voices": {
            "google": [
                {"value": "Charon", "label": "Charon"},
                {"value": "Kore", "label": "Kore"},
                {"value": "Fenrir", "label": "Fenrir"},
                {"value": "Aoede", "label": "Aoede"},
                {"value": "Puck", "label": "Puck"},
                {"value": "Leda", "label": "Leda"},
                {"value": "Orus", "label": "Orus"},
                {"value": "Zephyr", "label": "Zephyr"},
            ],
            "openai": [
                {"value": "alloy", "label": "Alloy"},
                {"value": "echo", "label": "Echo"},
                {"value": "fable", "label": "Fable"},
                {"value": "onyx", "label": "Onyx"},
                {"value": "nova", "label": "Nova"},
                {"value": "shimmer", "label": "Shimmer"},
                {"value": "ash", "label": "Ash"},
                {"value": "coral", "label": "Coral"},
                {"value": "sage", "label": "Sage"},
            ],
        },
        "image_providers": [
            {"value": "gemini", "label": "Google Gemini"},
            {"value": "openai", "label": "OpenAI"},
            {"value": "togetherai", "label": "Together AI"},
        ],
        "image_models": {
            "gemini": [
                {"value": "gemini-3.1-flash-image-preview", "label": "Gemini 3.1 Flash Image Preview"},
                {"value": "gemini-2.0-flash-preview-image-generation", "label": "Gemini 2.0 Flash Image Gen"},
                {"value": "imagen-3.0-generate-002", "label": "Imagen 3.0"},
            ],
            "openai": [
                {"value": "gpt-image-1-mini", "label": "GPT Image 1 Mini"},
                {"value": "gpt-image-1", "label": "GPT Image 1"},
                {"value": "gpt-image-1.5", "label": "GPT Image 1.5"},
            ],
            "togetherai": [
                {"value": "black-forest-labs/FLUX.1-schnell", "label": "FLUX.1 Schnell"},
                {"value": "black-forest-labs/FLUX.1-dev", "label": "FLUX.1 Dev"},
                {"value": "black-forest-labs/FLUX.1.1-pro", "label": "FLUX.1.1 Pro"},
                {"value": "stabilityai/stable-diffusion-xl-base-1.0", "label": "Stable Diffusion XL"},
                {"value": "Lykon/dreamshaper-xl-v2-turbo", "label": "DreamShaper XL v2 Turbo"},
            ],
        },
        "openai_image_sizes": [
            {"value": "1024x1024", "label": "1024×1024 (Square)"},
            {"value": "1536x1024", "label": "1536×1024 (Landscape)"},
            {"value": "1024x1536", "label": "1024×1536 (Portrait)"},
            {"value": "auto", "label": "Auto"},
        ],
        "togetherai_sizes": [
            {"value": "1024x576", "label": "1024×576 (Widescreen)"},
            {"value": "1280x720", "label": "1280×720 (HD)"},
            {"value": "1024x1024", "label": "1024×1024 (Square)"},
            {"value": "768x1024", "label": "768×1024 (Portrait)"},
        ],
        "resolutions": [
            {"value": "480p", "label": "480p (854×480)"},
            {"value": "720p", "label": "720p (1280×720)"},
            {"value": "1080p", "label": "1080p (1920×1080)"},
            {"value": "1440p", "label": "1440p (2560×1440)"},
            {"value": "4K", "label": "4K (3840×2160)"},
        ],
        "aspect_ratios": [
            {"value": "16:9", "label": "16:9 (Widescreen)"},
            {"value": "9:16", "label": "9:16 (Vertical/Mobile)"},
            {"value": "1:1", "label": "1:1 (Square)"},
            {"value": "4:3", "label": "4:3 (Standard)"},
        ],
    }


@app.get("/api/default-scenes")
async def get_default_scenes():
    from script_content import scenes
    return {"scenes": scenes}


# Serve React frontend (must be last)
frontend_build = Path(__file__).parent / "frontend" / "dist"
if frontend_build.exists():
    app.mount("/", StaticFiles(directory=str(frontend_build), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
