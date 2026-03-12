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
    image_model: str = "gemini-3.1-flash-image-preview"
    aspect_ratio: str = "16:9"
    image_size: str = "512"
    resolution: str = "1080p"

class TestAudioRequest(BaseModel):
    text: str
    speech_provider: str = "google"
    speech_model: str = "gemini-2.5-pro-preview-tts"
    speech_voice: str = "Charon"

class TestImageRequest(BaseModel):
    prompt: str
    image_model: str = "gemini-3.1-flash-image-preview"
    aspect_ratio: str = "16:9"
    image_size: str = "512"


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
            
            # Generate audio
            job["message"] = f"Scene {index + 1}/{total_scenes}: Generating audio..."
            if request.speech_provider == "openai":
                from openai_services import generate_audio_openai
                generate_audio_openai(scene.voiceover, audio_path, model=request.speech_model, voice=request.speech_voice)
            else:
                from gemini_services import generate_audio
                generate_audio(scene.voiceover, audio_path, model=request.speech_model, voice=request.speech_voice)
            
            # Generate image
            job["message"] = f"Scene {index + 1}/{total_scenes}: Generating image..."
            from gemini_services import generate_image
            generate_image(scene.prompt, image_path, model=request.image_model, aspect_ratio=request.aspect_ratio, image_size=request.image_size)
            
            job["progress"] = int(((index + 1) / total_scenes) * 90)
        
        # Assemble video
        job["progress"] = 92
        job["message"] = "Assembling final video..."
        from video_editor import assemble_final_video
        output_path = assemble_final_video(total_scenes, job_dir, "output.mp4", resolution=request.resolution)
        
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
        job["progress"] = 0


@app.post("/api/generate-video")
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued...",
        "output_path": None,
    }
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
        from gemini_services import generate_image
        success = generate_image(request.prompt, image_path, model=request.image_model, aspect_ratio=request.aspect_ratio, image_size=request.image_size)
        
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
        "image_models": [
            {"value": "gemini-3.1-flash-image-preview", "label": "Gemini 3.1 Flash Image Preview"},
            {"value": "gemini-2.0-flash-preview-image-generation", "label": "Gemini 2.0 Flash Image Gen"},
            {"value": "imagen-3.0-generate-002", "label": "Imagen 3.0"},
        ],
        "resolutions": [
            {"value": "480p", "label": "480p (854x480)"},
            {"value": "720p", "label": "720p (1280x720)"},
            {"value": "1080p", "label": "1080p (1920x1080)"},
            {"value": "1440p", "label": "1440p (2560x1440)"},
            {"value": "4K", "label": "4K (3840x2160)"},
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
