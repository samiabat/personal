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


class VisualBeat(BaseModel):
    trigger_word: str
    effect: str  # zoom_in_slow, zoom_out_slow, audio_reactive_shake, hard_cut, pop_scale
    image_index: int
    color_grade: Optional[str] = None
    focus_x: Optional[float] = None
    focus_y: Optional[float] = None


class V2Scene(BaseModel):
    voiceover: str
    prompts: list[str]
    visual_beats: list[VisualBeat]


class VideoRequest(BaseModel):
    version: str = "v1"  # "v1", "v2", or "v3"
    # V1 fields
    scenes: list[Scene] = []
    # V2/V3 fields
    v2_scenes: list[V2Scene] = []
    # Common fields
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
    orientation: str = "landscape"
    enable_ken_burns: bool = False
    enable_zoom: bool = False
    enable_shake: bool = False
    enable_subtitles: bool = False
    subtitle_style: str = "cinematic"  # "cinematic", "minimal", or "typewriter"
    gemini_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""
    together_api_key: Optional[str] = ""

class TestAudioRequest(BaseModel):
    text: str
    speech_provider: str = "google"
    speech_model: str = "gemini-2.5-pro-preview-tts"
    speech_voice: str = "Charon"
    gemini_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""

class TestImageRequest(BaseModel):
    prompt: str
    image_provider: str = "gemini"
    image_model: str = "gemini-3.1-flash-image-preview"
    aspect_ratio: str = "16:9"
    image_size: str = "512"
    openai_image_size: str = "1024x1024"
    togetherai_width: int = 1024
    togetherai_height: int = 576
    gemini_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""
    together_api_key: Optional[str] = ""


def _generate_image_for_provider(provider: str, prompt: str, image_path: str, **kwargs):
    """Route image generation to the correct provider."""
    if provider == "openai":
        from image_services import generate_image_openai
        return generate_image_openai(
            prompt, image_path,
            model=kwargs.get("image_model", "gpt-image-1"),
            size=kwargs.get("openai_image_size", "1024x1024"),
            api_key=kwargs.get("openai_api_key", ""),
        )
    elif provider == "togetherai":
        from image_services import generate_image_togetherai
        return generate_image_togetherai(
            prompt, image_path,
            model=kwargs.get("image_model", "black-forest-labs/FLUX.1-schnell"),
            width=kwargs.get("togetherai_width", 1024),
            height=kwargs.get("togetherai_height", 576),
            api_key=kwargs.get("together_api_key", ""),
        )
    else:  # gemini
        from gemini_services import generate_image
        return generate_image(
            prompt, image_path,
            model=kwargs.get("image_model", "gemini-3.1-flash-image-preview"),
            aspect_ratio=kwargs.get("aspect_ratio", "16:9"),
            image_size=kwargs.get("image_size", "512"),
            api_key=kwargs.get("gemini_api_key", ""),
        )


def run_asset_generation(job_id: str, request: VideoRequest):
    """Background task to generate audio + images only (no video assembly)."""
    job = jobs[job_id]
    job_dir = f"{JOBS_DIR}/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    total_scenes = len(request.scenes)

    try:
        for index, scene in enumerate(request.scenes):
            progress = int((index / total_scenes) * 100)
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
                    generate_audio_openai(scene.voiceover, audio_path, model=request.speech_model, voice=request.speech_voice, api_key=request.openai_api_key or "")
                else:
                    from gemini_services import generate_audio
                    generate_audio(scene.voiceover, audio_path, model=request.speech_model, voice=request.speech_voice, api_key=request.gemini_api_key or "")

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
                    gemini_api_key=request.gemini_api_key or "",
                    openai_api_key=request.openai_api_key or "",
                    together_api_key=request.together_api_key or "",
                )

            job["progress"] = int(((index + 1) / total_scenes) * 100)

        job["progress"] = 100
        job["status"] = "assets_ready"
        job["message"] = "Audio and images generated! Ready for review."

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["progress"] = job.get("progress", 0)


def run_video_assembly(job_id: str, request: VideoRequest):
    """Background task to assemble video from existing assets."""
    job = jobs[job_id]
    job_dir = f"{JOBS_DIR}/{job_id}"
    total_scenes = len(request.scenes)

    try:
        job["progress"] = 10
        job["status"] = "processing"
        job["message"] = "Assembling final video..."
        from video_editor import assemble_final_video
        output_path = assemble_final_video(
            total_scenes, job_dir, "output.mp4",
            resolution=request.resolution,
            orientation=request.orientation,
            enable_ken_burns=request.enable_ken_burns,
            enable_zoom=request.enable_zoom,
            enable_shake=request.enable_shake,
        )

        # Clean up temp files
        job["progress"] = 90
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


def _get_v2_resolution(request: VideoRequest) -> tuple:
    """Resolve the target (width, height) for V2 assembly."""
    res_map = {
        "480p": (854, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4K": (3840, 2160),
    }
    size = res_map.get(request.resolution, (1920, 1080))
    if request.orientation == "portrait":
        size = (size[1], size[0])
    return size


def run_v2_asset_generation(job_id: str, request: VideoRequest):
    """Background task: generate audio + images for all V2 grouped scenes."""
    job = jobs[job_id]
    job_dir = f"{JOBS_DIR}/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    total_scenes = len(request.v2_scenes)

    try:
        for s_idx, scene in enumerate(request.v2_scenes):
            # --- Audio (one per grouped scene) ---
            audio_path = f"{job_dir}/v2_scene_{s_idx}_audio.wav"
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                job["message"] = f"V2 Scene {s_idx + 1}/{total_scenes}: Audio exists, skipping..."
            else:
                job["message"] = f"V2 Scene {s_idx + 1}/{total_scenes}: Generating voiceover..."
                job["status"] = "processing"
                if request.speech_provider == "openai":
                    from openai_services import generate_audio_openai
                    generate_audio_openai(scene.voiceover, audio_path,
                                          model=request.speech_model,
                                          voice=request.speech_voice,
                                          api_key=request.openai_api_key or "")
                else:
                    from gemini_services import generate_audio
                    generate_audio(scene.voiceover, audio_path,
                                   model=request.speech_model,
                                   voice=request.speech_voice,
                                   api_key=request.gemini_api_key or "")

            # --- Images (one per prompt) ---
            for p_idx, prompt in enumerate(scene.prompts):
                image_path = f"{job_dir}/v2_scene_{s_idx}_image_{p_idx}.jpg"
                if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                    job["message"] = f"V2 Scene {s_idx + 1}: Image {p_idx + 1}/{len(scene.prompts)} exists, skipping..."
                else:
                    job["message"] = f"V2 Scene {s_idx + 1}: Generating image {p_idx + 1}/{len(scene.prompts)}..."
                    _generate_image_for_provider(
                        request.image_provider, prompt, image_path,
                        image_model=request.image_model,
                        aspect_ratio=request.aspect_ratio,
                        image_size=request.image_size,
                        openai_image_size=request.openai_image_size,
                        togetherai_width=request.togetherai_width,
                        togetherai_height=request.togetherai_height,
                        gemini_api_key=request.gemini_api_key or "",
                        openai_api_key=request.openai_api_key or "",
                        together_api_key=request.together_api_key or "",
                    )

            job["progress"] = int(((s_idx + 1) / total_scenes) * 100)

        job["progress"] = 100
        job["status"] = "assets_ready"
        job["message"] = "V2 assets generated! Ready for review."

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["progress"] = job.get("progress", 0)


def run_v2_video_assembly(job_id: str, request: VideoRequest):
    """Background task: assemble V2 video using word-level timestamps + effects."""
    job = jobs[job_id]
    job_dir = f"{JOBS_DIR}/{job_id}"

    try:
        job["progress"] = 5
        job["status"] = "processing"
        job["message"] = "Analysing audio for word timestamps..."

        from v2_editor import get_word_timestamps, assemble_v2_video

        target_size = _get_v2_resolution(request)
        all_scene_clips_paths: list[str] = []

        for s_idx, scene in enumerate(request.v2_scenes):
            audio_path = f"{job_dir}/v2_scene_{s_idx}_audio.wav"
            image_paths = [
                f"{job_dir}/v2_scene_{s_idx}_image_{p}.jpg"
                for p in range(len(scene.prompts))
            ]

            # Get word-level timestamps via Whisper
            job["message"] = f"V2 Scene {s_idx + 1}: Extracting word timestamps..."
            word_ts = get_word_timestamps(audio_path, api_key=request.openai_api_key or "")

            # Assemble this grouped scene
            scene_output = f"{job_dir}/v2_scene_{s_idx}_output.mp4"
            job["message"] = f"V2 Scene {s_idx + 1}: Assembling video with effects..."
            beats = [b.model_dump() for b in scene.visual_beats]
            assemble_v2_video(image_paths, audio_path, beats, word_ts,
                              scene_output, target_size=target_size,
                              enable_subtitles=request.enable_subtitles,
                              subtitle_style=request.subtitle_style)
            all_scene_clips_paths.append(scene_output)

            job["progress"] = int(10 + (s_idx + 1) / len(request.v2_scenes) * 70)

        # If multiple grouped scenes, concatenate them
        if len(all_scene_clips_paths) == 1:
            final_output = f"{job_dir}/output.mp4"
            os.rename(all_scene_clips_paths[0], final_output)
        else:
            from moviepy import VideoFileClip, concatenate_videoclips
            job["message"] = "Concatenating grouped scenes..."
            clips = [VideoFileClip(p) for p in all_scene_clips_paths]
            final = concatenate_videoclips(clips, method="compose")
            final_output = f"{job_dir}/output.mp4"
            final.write_videofile(final_output, fps=24, codec="libx264", audio_codec="aac")

        # Clean up temp files
        job["progress"] = 90
        job["message"] = "Cleaning up temporary files..."
        for s_idx, scene in enumerate(request.v2_scenes):
            audio_path = f"{job_dir}/v2_scene_{s_idx}_audio.wav"
            if os.path.exists(audio_path):
                os.remove(audio_path)
            for p_idx in range(len(scene.prompts)):
                img_path = f"{job_dir}/v2_scene_{s_idx}_image_{p_idx}.jpg"
                if os.path.exists(img_path):
                    os.remove(img_path)
            scene_output = f"{job_dir}/v2_scene_{s_idx}_output.mp4"
            if os.path.exists(scene_output):
                os.remove(scene_output)

        job["progress"] = 100
        job["status"] = "completed"
        job["message"] = "V2 video generation completed!"
        job["output_path"] = final_output

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["progress"] = job.get("progress", 0)


def run_v2_video_generation(job_id: str, request: VideoRequest):
    """Full V2 pipeline: assets + assembly in one background task."""
    run_v2_asset_generation(job_id, request)
    job = jobs[job_id]
    if job["status"] == "assets_ready":
        job["status"] = "processing"
        job["progress"] = 0
        run_v2_video_assembly(job_id, request)


def run_v3_video_generation(job_id: str, request: VideoRequest):
    """Full V3 pipeline: V2 asset generation + V3 focus-aware assembly."""
    run_v2_asset_generation(job_id, request)
    job = jobs[job_id]
    if job["status"] == "assets_ready":
        job["status"] = "processing"
        job["progress"] = 0
        run_v3_video_assembly(job_id, request)


def run_v3_video_assembly(job_id: str, request: VideoRequest):
    """Background task: assemble V3 video using word-level timestamps + focus-aware effects."""
    job = jobs[job_id]
    job_dir = f"{JOBS_DIR}/{job_id}"

    try:
        job["progress"] = 5
        job["status"] = "processing"
        job["message"] = "Analysing audio for word timestamps..."

        from v2_editor import get_word_timestamps
        from v3_editor import assemble_v3_video

        target_size = _get_v2_resolution(request)
        all_scene_clips_paths: list[str] = []

        for s_idx, scene in enumerate(request.v2_scenes):
            audio_path = f"{job_dir}/v2_scene_{s_idx}_audio.wav"
            image_paths = [
                f"{job_dir}/v2_scene_{s_idx}_image_{p}.jpg"
                for p in range(len(scene.prompts))
            ]

            job["message"] = f"V3 Scene {s_idx + 1}: Extracting word timestamps..."
            word_ts = get_word_timestamps(audio_path, api_key=request.openai_api_key or "")

            scene_output = f"{job_dir}/v2_scene_{s_idx}_output.mp4"
            job["message"] = f"V3 Scene {s_idx + 1}: Assembling video with focus-aware effects..."
            beats = [b.model_dump() for b in scene.visual_beats]
            assemble_v3_video(image_paths, audio_path, beats, word_ts,
                              scene_output, target_size=target_size,
                              enable_subtitles=request.enable_subtitles,
                              subtitle_style=request.subtitle_style)
            all_scene_clips_paths.append(scene_output)

            job["progress"] = int(10 + (s_idx + 1) / len(request.v2_scenes) * 70)

        if len(all_scene_clips_paths) == 1:
            final_output = f"{job_dir}/output.mp4"
            os.rename(all_scene_clips_paths[0], final_output)
        else:
            from moviepy import VideoFileClip, concatenate_videoclips
            job["message"] = "Concatenating grouped scenes..."
            clips = [VideoFileClip(p) for p in all_scene_clips_paths]
            final = concatenate_videoclips(clips, method="compose")
            final_output = f"{job_dir}/output.mp4"
            final.write_videofile(final_output, fps=24, codec="libx264", audio_codec="aac")

        job["progress"] = 90
        job["message"] = "Cleaning up temporary files..."
        for s_idx, scene in enumerate(request.v2_scenes):
            audio_path = f"{job_dir}/v2_scene_{s_idx}_audio.wav"
            if os.path.exists(audio_path):
                os.remove(audio_path)
            for p_idx in range(len(scene.prompts)):
                img_path = f"{job_dir}/v2_scene_{s_idx}_image_{p_idx}.jpg"
                if os.path.exists(img_path):
                    os.remove(img_path)
            scene_output = f"{job_dir}/v2_scene_{s_idx}_output.mp4"
            if os.path.exists(scene_output):
                os.remove(scene_output)

        job["progress"] = 100
        job["status"] = "completed"
        job["message"] = "V3 video generation completed!"
        job["output_path"] = final_output

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["progress"] = job.get("progress", 0)


def run_video_generation(job_id: str, request: VideoRequest):
    """Background task to generate video (full pipeline: assets + assembly)."""
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
                    generate_audio_openai(scene.voiceover, audio_path, model=request.speech_model, voice=request.speech_voice, api_key=request.openai_api_key or "")
                else:
                    from gemini_services import generate_audio
                    generate_audio(scene.voiceover, audio_path, model=request.speech_model, voice=request.speech_voice, api_key=request.gemini_api_key or "")

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
                    gemini_api_key=request.gemini_api_key or "",
                    openai_api_key=request.openai_api_key or "",
                    together_api_key=request.together_api_key or "",
                )

            job["progress"] = int(((index + 1) / total_scenes) * 90)

        # Assemble video
        job["progress"] = 92
        job["message"] = "Assembling final video..."
        from video_editor import assemble_final_video
        output_path = assemble_final_video(
            total_scenes, job_dir, "output.mp4",
            resolution=request.resolution,
            orientation=request.orientation,
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
    if request.version == "v2":
        background_tasks.add_task(run_v2_video_generation, job_id, request)
    elif request.version == "v3":
        background_tasks.add_task(run_v3_video_generation, job_id, request)
    else:
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

    if request.version == "v3":
        background_tasks.add_task(run_v3_video_generation, job_id, request)
    elif request.version == "v2":
        background_tasks.add_task(run_v2_video_generation, job_id, request)
    else:
        background_tasks.add_task(run_video_generation, job_id, request)
    return {"job_id": job_id}


@app.post("/api/generate-assets")
async def generate_assets(request: VideoRequest, background_tasks: BackgroundTasks):
    """Generate audio and images only (no video assembly). Returns job_id for tracking."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued...",
        "output_path": None,
        "request": request.model_dump(),
    }
    if request.version in ("v2", "v3"):
        background_tasks.add_task(run_v2_asset_generation, job_id, request)
    else:
        background_tasks.add_task(run_asset_generation, job_id, request)
    return {"job_id": job_id}


@app.get("/api/job-assets/{job_id}")
async def get_job_assets(job_id: str):
    """Return list of generated assets for a job so user can review them."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] not in ("assets_ready", "approved", "completed"):
        raise HTTPException(status_code=400, detail=f"Assets not ready (status: {job['status']})")

    request_data = job["request"]
    job_dir = f"{JOBS_DIR}/{job_id}"

    version = request_data.get("version", "v1")

    if version in ("v2", "v3"):
        assets = []
        for s_idx, scene in enumerate(request_data.get("v2_scenes", [])):
            audio_path = f"{job_dir}/v2_scene_{s_idx}_audio.wav"
            scene_assets = {
                "scene_index": s_idx,
                "voiceover": scene["voiceover"],
                "has_audio": os.path.exists(audio_path) and os.path.getsize(audio_path) > 0,
                "audio_url": f"/api/scene-audio/{job_id}/{s_idx}",
                "images": [],
                "visual_beats": scene.get("visual_beats", []),
            }
            for p_idx, prompt in enumerate(scene["prompts"]):
                img_path = f"{job_dir}/v2_scene_{s_idx}_image_{p_idx}.jpg"
                scene_assets["images"].append({
                    "prompt_index": p_idx,
                    "prompt": prompt,
                    "has_image": os.path.exists(img_path) and os.path.getsize(img_path) > 0,
                    "image_url": f"/api/v2-scene-image/{job_id}/{s_idx}/{p_idx}",
                })
            assets.append(scene_assets)
        total = len(request_data.get("v2_scenes", []))
        return {"job_id": job_id, "version": version, "total_scenes": total, "assets": assets}

    # V1 path
    total_scenes = len(request_data["scenes"])

    assets = []
    for i in range(total_scenes):
        image_path = f"{job_dir}/scene_{i}_image.jpg"
        audio_path = f"{job_dir}/scene_{i}_audio.wav"
        assets.append({
            "scene_index": i,
            "prompt": request_data["scenes"][i]["prompt"],
            "voiceover": request_data["scenes"][i]["voiceover"],
            "has_image": os.path.exists(image_path) and os.path.getsize(image_path) > 0,
            "has_audio": os.path.exists(audio_path) and os.path.getsize(audio_path) > 0,
            "image_url": f"/api/scene-image/{job_id}/{i}",
            "audio_url": f"/api/scene-audio/{job_id}/{i}",
        })

    return {"job_id": job_id, "version": "v1", "total_scenes": total_scenes, "assets": assets}


@app.get("/api/scene-image/{job_id}/{scene_index}")
async def get_scene_image(job_id: str, scene_index: int):
    """Serve a specific scene image for review."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    image_path = f"{JOBS_DIR}/{job_id}/scene_{scene_index}_image.jpg"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        image_path, media_type="image/jpeg",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.get("/api/v2-scene-image/{job_id}/{scene_index}/{prompt_index}")
async def get_v2_scene_image(job_id: str, scene_index: int, prompt_index: int):
    """Serve a specific V2 scene image for review."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    image_path = f"{JOBS_DIR}/{job_id}/v2_scene_{scene_index}_image_{prompt_index}.jpg"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        image_path, media_type="image/jpeg",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.get("/api/scene-audio/{job_id}/{scene_index}")
async def get_scene_audio(job_id: str, scene_index: int):
    """Serve a specific scene audio for review (V1, V2, or V3)."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    version = job["request"].get("version", "v1")
    if version in ("v2", "v3"):
        audio_path = f"{JOBS_DIR}/{job_id}/v2_scene_{scene_index}_audio.wav"
    else:
        audio_path = f"{JOBS_DIR}/{job_id}/scene_{scene_index}_audio.wav"
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(audio_path, media_type="audio/wav")


@app.post("/api/regenerate-image/{job_id}/{scene_index}")
async def regenerate_image(job_id: str, scene_index: int):
    """Regenerate a specific scene's image."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "assets_ready":
        raise HTTPException(status_code=400, detail="Assets not ready for regeneration")

    request_data = job["request"]
    scenes = request_data["scenes"]
    if scene_index < 0 or scene_index >= len(scenes):
        raise HTTPException(status_code=400, detail="Invalid scene index")

    image_path = f"{JOBS_DIR}/{job_id}/scene_{scene_index}_image.jpg"

    # Delete existing image
    if os.path.exists(image_path):
        os.remove(image_path)

    try:
        _generate_image_for_provider(
            request_data.get("image_provider", "gemini"),
            scenes[scene_index]["prompt"],
            image_path,
            image_model=request_data.get("image_model", "gemini-3.1-flash-image-preview"),
            aspect_ratio=request_data.get("aspect_ratio", "16:9"),
            image_size=request_data.get("image_size", "512"),
            openai_image_size=request_data.get("openai_image_size", "1024x1024"),
            togetherai_width=request_data.get("togetherai_width", 1024),
            togetherai_height=request_data.get("togetherai_height", 576),
            gemini_api_key=request_data.get("gemini_api_key", ""),
            openai_api_key=request_data.get("openai_api_key", ""),
            together_api_key=request_data.get("together_api_key", ""),
        )
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            return {"success": True, "message": f"Image for scene {scene_index + 1} regenerated"}
        raise HTTPException(status_code=500, detail="Image regeneration failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/regenerate-v2-image/{job_id}/{scene_index}/{prompt_index}")
async def regenerate_v2_image(job_id: str, scene_index: int, prompt_index: int):
    """Regenerate a specific V2 scene image."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "assets_ready":
        raise HTTPException(status_code=400, detail="Assets not ready for regeneration")

    request_data = job["request"]
    v2_scenes = request_data.get("v2_scenes", [])
    if scene_index < 0 or scene_index >= len(v2_scenes):
        raise HTTPException(status_code=400, detail="Invalid scene index")
    prompts = v2_scenes[scene_index].get("prompts", [])
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise HTTPException(status_code=400, detail="Invalid prompt index")

    image_path = f"{JOBS_DIR}/{job_id}/v2_scene_{scene_index}_image_{prompt_index}.jpg"
    if os.path.exists(image_path):
        os.remove(image_path)

    try:
        _generate_image_for_provider(
            request_data.get("image_provider", "gemini"),
            prompts[prompt_index],
            image_path,
            image_model=request_data.get("image_model", "gemini-3.1-flash-image-preview"),
            aspect_ratio=request_data.get("aspect_ratio", "16:9"),
            image_size=request_data.get("image_size", "512"),
            openai_image_size=request_data.get("openai_image_size", "1024x1024"),
            togetherai_width=request_data.get("togetherai_width", 1024),
            togetherai_height=request_data.get("togetherai_height", 576),
            gemini_api_key=request_data.get("gemini_api_key", ""),
            openai_api_key=request_data.get("openai_api_key", ""),
            together_api_key=request_data.get("together_api_key", ""),
        )
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            return {"success": True, "message": f"V2 image {prompt_index + 1} for scene {scene_index + 1} regenerated"}
        raise HTTPException(status_code=500, detail="Image regeneration failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/approve-assets/{job_id}")
async def approve_assets(job_id: str):
    """Mark assets as approved, enabling video assembly."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "assets_ready":
        raise HTTPException(status_code=400, detail="Assets not in reviewable state")
    job["status"] = "approved"
    job["message"] = "Assets approved. Ready to prepare video."
    return {"success": True}


class FocusPointUpdate(BaseModel):
    scene_index: int
    beat_index: int
    focus_x: float
    focus_y: float


@app.post("/api/update-focus-point/{job_id}")
async def update_focus_point(job_id: str, update: FocusPointUpdate):
    """Save a director-selected focus point into a V3 visual beat."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]

    v2_scenes = job["request"].get("v2_scenes", [])
    if update.scene_index < 0 or update.scene_index >= len(v2_scenes):
        raise HTTPException(status_code=400, detail="Invalid scene index")
    beats = v2_scenes[update.scene_index].get("visual_beats", [])
    if update.beat_index < 0 or update.beat_index >= len(beats):
        raise HTTPException(status_code=400, detail="Invalid beat index")

    # Clamp to [0, 1]
    fx = max(0.0, min(1.0, update.focus_x))
    fy = max(0.0, min(1.0, update.focus_y))

    beats[update.beat_index]["focus_x"] = fx
    beats[update.beat_index]["focus_y"] = fy
    return {"success": True, "focus_x": fx, "focus_y": fy}


@app.post("/api/prepare-video/{job_id}")
async def prepare_video(job_id: str, background_tasks: BackgroundTasks):
    """Assemble video from approved assets."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "approved":
        raise HTTPException(status_code=400, detail="Assets must be approved before video preparation")

    request = VideoRequest(**job["request"])

    # Reset progress for video assembly phase
    job["status"] = "queued"
    job["progress"] = 0
    job["message"] = "Preparing video..."
    job["output_path"] = None

    if request.version == "v3":
        background_tasks.add_task(run_v3_video_assembly, job_id, request)
    elif request.version == "v2":
        background_tasks.add_task(run_v2_video_assembly, job_id, request)
    else:
        background_tasks.add_task(run_video_assembly, job_id, request)
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
            if job["status"] in ("completed", "failed", "assets_ready"):
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
            success = generate_audio_openai(request.text, audio_path, model=request.speech_model, voice=request.speech_voice, api_key=request.openai_api_key or "")
        else:
            from gemini_services import generate_audio
            success = generate_audio(request.text, audio_path, model=request.speech_model, voice=request.speech_voice, api_key=request.gemini_api_key or "")

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
            gemini_api_key=request.gemini_api_key or "",
            openai_api_key=request.openai_api_key or "",
            together_api_key=request.together_api_key or "",
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
            {"value": "480p", "label": "480p"},
            {"value": "720p", "label": "720p"},
            {"value": "1080p", "label": "1080p"},
            {"value": "1440p", "label": "1440p"},
            {"value": "4K", "label": "4K"},
        ],
        "orientations": [
            {"value": "landscape", "label": "Landscape (16:9)"},
            {"value": "portrait", "label": "Portrait (9:16)"},
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
