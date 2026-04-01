import os
import uuid
import json
import asyncio
import shutil
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sse_starlette.sse import EventSourceResponse
from PIL import Image as PILImage
import io

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


def _job_entry_from_disk(job_id: str, request_data: dict) -> dict:
    """Build a job dict from disk state (used on startup and in _ensure_job_in_memory)."""
    status, output_path = _detect_disk_status(job_id, request_data)
    progress = 100 if status == "completed" else (80 if status == "assets_ready" else 0)
    message = {
        "completed": "Loaded from disk -- video ready.",
        "assets_ready": "Loaded from disk -- assets ready for review.",
        "failed": "Loaded from disk -- previous run did not finish.",
    }.get(status, "Loaded from disk.")
    return {
        "id": job_id,
        "status": status,
        "progress": progress,
        "message": message,
        "output_path": output_path,
        "request": request_data,
    }


@app.on_event("startup")
async def _load_all_jobs_from_disk():
    """On server start, scan the jobs directory and load all existing jobs into memory
    so they are immediately accessible without requiring manual recovery."""
    if not os.path.isdir(JOBS_DIR):
        return
    for job_id in os.listdir(JOBS_DIR):
        if job_id in jobs:
            continue
        job_dir = f"{JOBS_DIR}/{job_id}"
        request_path = f"{job_dir}/request.json"
        if not os.path.isdir(job_dir) or not os.path.exists(request_path):
            continue
        try:
            with open(request_path) as f:
                request_data = json.load(f)
            jobs[job_id] = _job_entry_from_disk(job_id, request_data)
        except Exception as exc:
            print(f"[startup] Skipping job {job_id}: {exc}")


def _ensure_job_in_memory(job_id: str) -> dict:
    """Return in-memory job dict, auto-loading from disk if not present.

    Raises HTTPException(404) if the job cannot be found on disk either.
    """
    if job_id in jobs:
        return jobs[job_id]
    job_dir = f"{JOBS_DIR}/{job_id}"
    request_path = f"{job_dir}/request.json"
    if not os.path.isdir(job_dir) or not os.path.exists(request_path):
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        with open(request_path) as f:
            request_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read job metadata: {e}")
    jobs[job_id] = _job_entry_from_disk(job_id, request_data)
    return jobs[job_id]


def _save_job_metadata(job_id: str, request_data: dict):
    """Persist job request data to disk so it can be resumed after server restarts."""
    job_dir = f"{JOBS_DIR}/{job_id}"
    os.makedirs(job_dir, exist_ok=True)
    try:
        with open(f"{job_dir}/request.json", "w") as f:
            json.dump(request_data, f)
    except Exception:
        pass  # Non-fatal: in-memory job still works


def _detect_disk_status(job_id: str, request_data: dict) -> tuple[str, str | None]:
    """Inspect files on disk and return (status, output_path)."""
    job_dir = f"{JOBS_DIR}/{job_id}"
    output_path = f"{job_dir}/output.mp4"
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return "completed", output_path
    # Check if any assets were generated
    version = request_data.get("version", "v1")
    if version == "v5":
        audio_0 = f"{job_dir}/v5_scene_0_audio.wav"
    elif version == "v6":
        audio_0 = f"{job_dir}/v6_scene_0_audio.wav"
    elif version in ("v2", "v3"):
        audio_0 = f"{job_dir}/v2_scene_0_audio.wav"
    else:
        audio_0 = f"{job_dir}/scene_0_audio.wav"
    if os.path.exists(audio_0) and os.path.getsize(audio_0) > 0:
        return "assets_ready", None
    return "failed", None

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


class V5Scene(BaseModel):
    voiceover: str
    prompt: str
    media_type: str = "video"
    time_fit_strategy: str = "auto"  # "auto", "trim", "cinematic_slow_mo", "loop_or_freeze"


class V6Scene(BaseModel):
    voiceover: str
    prompt: str
    media_type: str = "image"          # "image" or "video"
    # Image-scene options
    zoom_effect: str = "zoom_in"       # "none", "zoom_in", "zoom_out", "ken_burns"
    focus_x: float = 0.5              # zoom focus point (0–1)
    focus_y: float = 0.5
    # Video-scene options
    time_fit_strategy: str = "auto"   # "auto", "trim", "cinematic_slow_mo", "loop_or_freeze"


class VideoRequest(BaseModel):
    version: str = "v1"  # "v1", "v2", "v3", "v5", or "v6"
    # V1 fields
    scenes: list[Scene] = []
    # V2/V3 fields
    v2_scenes: list[V2Scene] = []
    # V5 fields
    v5_scenes: list[V5Scene] = []
    # V6 fields
    v6_scenes: list[V6Scene] = []
    # Common fields
    speech_provider: str = "google"  # "google", "openai", or "elevenlabs"
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
    elevenlabs_api_key: Optional[str] = ""

class TestAudioRequest(BaseModel):
    text: str
    speech_provider: str = "google"
    speech_model: str = "gemini-2.5-pro-preview-tts"
    speech_voice: str = "Charon"
    gemini_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""
    elevenlabs_api_key: Optional[str] = ""

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


def _generate_audio_for_provider(provider: str, text: str, audio_path: str, **kwargs):
    """Route TTS audio generation to the correct provider."""
    if provider == "openai":
        from openai_services import generate_audio_openai
        return generate_audio_openai(
            text, audio_path,
            model=kwargs.get("speech_model", "tts-1"),
            voice=kwargs.get("speech_voice", "alloy"),
            api_key=kwargs.get("openai_api_key", ""),
        )
    elif provider == "elevenlabs":
        from elevenlabs_services import generate_audio_elevenlabs
        return generate_audio_elevenlabs(
            text, audio_path,
            model=kwargs.get("speech_model", "eleven_multilingual_v2"),
            voice=kwargs.get("speech_voice", "3TStB8f3X3To0Uj5R7RK"),
            api_key=kwargs.get("elevenlabs_api_key", ""),
        )
    else:  # google
        from gemini_services import generate_audio
        return generate_audio(
            text, audio_path,
            model=kwargs.get("speech_model", "gemini-2.5-pro-preview-tts"),
            voice=kwargs.get("speech_voice", "Charon"),
            api_key=kwargs.get("gemini_api_key", ""),
        )


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
                _generate_audio_for_provider(
                    request.speech_provider, scene.voiceover, audio_path,
                    speech_model=request.speech_model,
                    speech_voice=request.speech_voice,
                    gemini_api_key=request.gemini_api_key or "",
                    openai_api_key=request.openai_api_key or "",
                    elevenlabs_api_key=request.elevenlabs_api_key or "",
                )

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
                _generate_audio_for_provider(
                    request.speech_provider, scene.voiceover, audio_path,
                    speech_model=request.speech_model,
                    speech_voice=request.speech_voice,
                    gemini_api_key=request.gemini_api_key or "",
                    openai_api_key=request.openai_api_key or "",
                    elevenlabs_api_key=request.elevenlabs_api_key or "",
                )

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

        # Clean up only intermediate per-scene output files (not audio/images — kept for re-use)
        job["progress"] = 90
        job["message"] = "Finalising video..."
        for s_idx in range(len(request.v2_scenes)):
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

        # Clean up only intermediate per-scene output files (not audio/images — kept for re-use)
        job["progress"] = 90
        job["message"] = "Finalising video..."
        for s_idx in range(len(request.v2_scenes)):
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
                _generate_audio_for_provider(
                    request.speech_provider, scene.voiceover, audio_path,
                    speech_model=request.speech_model,
                    speech_voice=request.speech_voice,
                    gemini_api_key=request.gemini_api_key or "",
                    openai_api_key=request.openai_api_key or "",
                    elevenlabs_api_key=request.elevenlabs_api_key or "",
                )

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

        job["progress"] = 100
        job["status"] = "completed"
        job["message"] = "Video generation completed!"
        job["output_path"] = output_path

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["progress"] = job.get("progress", 0)


# ── V5 pipeline ─────────────────────────────────────────────────────────────

def run_v5_tts_generation(job_id: str, request: VideoRequest):
    """Background task: generate TTS audio for every V5 scene.

    Video clips are uploaded manually by the user via the Asset Dashboard,
    so this only produces the voiceover audio files.
    """
    job = jobs[job_id]
    job_dir = f"{JOBS_DIR}/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    total = len(request.v5_scenes)

    try:
        for idx, scene in enumerate(request.v5_scenes):
            audio_path = f"{job_dir}/v5_scene_{idx}_audio.wav"
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                job["message"] = f"V5 Scene {idx + 1}/{total}: Audio exists, skipping..."
            else:
                job["message"] = f"V5 Scene {idx + 1}/{total}: Generating voiceover..."
                job["status"] = "processing"
                _generate_audio_for_provider(
                    request.speech_provider, scene.voiceover, audio_path,
                    speech_model=request.speech_model,
                    speech_voice=request.speech_voice,
                    gemini_api_key=request.gemini_api_key or "",
                    openai_api_key=request.openai_api_key or "",
                    elevenlabs_api_key=request.elevenlabs_api_key or "",
                )
            job["progress"] = int(((idx + 1) / total) * 100)

        job["progress"] = 100
        job["status"] = "assets_ready"
        job["message"] = "V5 voiceover audio generated! Upload video clips to proceed."

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["progress"] = job.get("progress", 0)


def run_v5_video_assembly(job_id: str, request: VideoRequest):
    """Background task: assemble V5 video from uploaded clips + TTS audio."""
    job = jobs[job_id]
    job_dir = f"{JOBS_DIR}/{job_id}"

    try:
        job["progress"] = 5
        job["status"] = "processing"
        job["message"] = "V5: Assembling video with time-remapping..."

        from v5_editor import assemble_v5_video

        scene_dicts = [s.model_dump() for s in request.v5_scenes]
        output_path = assemble_v5_video(
            scene_dicts,
            job_dir,
            output_filename="output.mp4",
            resolution=request.resolution,
            orientation=request.orientation,
            enable_subtitles=request.enable_subtitles,
            subtitle_style=request.subtitle_style,
        )

        job["progress"] = 100
        job["status"] = "completed"
        job["message"] = "V5 video generation completed!"
        job["output_path"] = output_path

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["progress"] = job.get("progress", 0)


# ── V6 pipeline ─────────────────────────────────────────────────────────────

def run_v6_asset_generation(job_id: str, request: VideoRequest):
    """Background task: generate TTS audio + AI images for every V6 scene.

    For *image* scenes both audio and an AI image are generated.
    For *video* scenes only the TTS audio is generated; the user uploads
    the video clip via the Asset Dashboard.
    """
    job = jobs[job_id]
    job_dir = f"{JOBS_DIR}/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    total = len(request.v6_scenes)

    try:
        for idx, scene in enumerate(request.v6_scenes):
            job["status"] = "processing"

            # ── TTS audio ──
            audio_path = f"{job_dir}/v6_scene_{idx}_audio.wav"
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                job["message"] = f"V6 Scene {idx + 1}/{total}: Audio exists, skipping..."
            else:
                job["message"] = f"V6 Scene {idx + 1}/{total}: Generating voiceover..."
                _generate_audio_for_provider(
                    request.speech_provider, scene.voiceover, audio_path,
                    speech_model=request.speech_model,
                    speech_voice=request.speech_voice,
                    gemini_api_key=request.gemini_api_key or "",
                    openai_api_key=request.openai_api_key or "",
                    elevenlabs_api_key=request.elevenlabs_api_key or "",
                )

            # ── AI image (image scenes only) ──
            if scene.media_type == "image":
                image_path = f"{job_dir}/v6_scene_{idx}_image.jpg"
                if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                    job["message"] = f"V6 Scene {idx + 1}/{total}: Image exists, skipping..."
                else:
                    job["message"] = f"V6 Scene {idx + 1}/{total}: Generating image..."
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

            job["progress"] = int(((idx + 1) / total) * 100)

        job["progress"] = 100
        job["status"] = "assets_ready"
        job["message"] = "V6 assets generated! Upload video clips for video scenes, then continue."

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["progress"] = job.get("progress", 0)


def run_v6_video_assembly(job_id: str, request: VideoRequest):
    """Background task: assemble V6 video from image/video scenes + TTS audio."""
    job = jobs[job_id]
    job_dir = f"{JOBS_DIR}/{job_id}"

    try:
        job["progress"] = 5
        job["status"] = "processing"
        job["message"] = "V6: Assembling hybrid video..."

        from v6_editor import assemble_v6_video

        scene_dicts = [s.model_dump() for s in request.v6_scenes]
        output_path = assemble_v6_video(
            scene_dicts,
            job_dir,
            output_filename="output.mp4",
            resolution=request.resolution,
            orientation=request.orientation,
            enable_subtitles=request.enable_subtitles,
            subtitle_style=request.subtitle_style,
        )

        job["progress"] = 100
        job["status"] = "completed"
        job["message"] = "V6 video generation completed!"
        job["output_path"] = output_path

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["progress"] = job.get("progress", 0)


@app.post("/api/generate-video")
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    request_data = request.model_dump()
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued...",
        "output_path": None,
        "request": request_data,
    }
    _save_job_metadata(job_id, request_data)
    if request.version == "v2":
        background_tasks.add_task(run_v2_video_generation, job_id, request)
    elif request.version == "v3":
        background_tasks.add_task(run_v3_video_generation, job_id, request)
    elif request.version == "v5":
        background_tasks.add_task(run_v5_tts_generation, job_id, request)
    elif request.version == "v6":
        background_tasks.add_task(run_v6_asset_generation, job_id, request)
    else:
        background_tasks.add_task(run_video_generation, job_id, request)
    return {"job_id": job_id}


@app.post("/api/retry/{job_id}")
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    """Retry a failed job, resuming from where it left off (skipping existing assets)."""
    job = _ensure_job_in_memory(job_id)
    if job["status"] != "failed":
        raise HTTPException(status_code=400, detail="Only failed jobs can be retried")

    # Rebuild the request from stored data
    request = VideoRequest(**job["request"])

    # Reset job status
    job["status"] = "queued"
    job["progress"] = 0
    job["message"] = "Retrying job..."
    job["output_path"] = None

    if request.version == "v5":
        background_tasks.add_task(run_v5_tts_generation, job_id, request)
    elif request.version == "v6":
        background_tasks.add_task(run_v6_asset_generation, job_id, request)
    elif request.version == "v3":
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
    request_data = request.model_dump()
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued...",
        "output_path": None,
        "request": request_data,
    }
    _save_job_metadata(job_id, request_data)
    if request.version == "v5":
        background_tasks.add_task(run_v5_tts_generation, job_id, request)
    elif request.version == "v6":
        background_tasks.add_task(run_v6_asset_generation, job_id, request)
    elif request.version in ("v2", "v3"):
        background_tasks.add_task(run_v2_asset_generation, job_id, request)
    else:
        background_tasks.add_task(run_asset_generation, job_id, request)
    return {"job_id": job_id}


@app.get("/api/job-assets/{job_id}")
async def get_job_assets(job_id: str):
    """Return list of generated assets for a job so user can review them."""
    job = _ensure_job_in_memory(job_id)
    if job["status"] not in ("assets_ready", "approved", "completed"):
        raise HTTPException(status_code=400, detail=f"Assets not ready (status: {job['status']})")

    request_data = job["request"]
    job_dir = f"{JOBS_DIR}/{job_id}"

    version = request_data.get("version", "v1")

    if version == "v6":
        assets = []
        for s_idx, scene in enumerate(request_data.get("v6_scenes", [])):
            audio_path = f"{job_dir}/v6_scene_{s_idx}_audio.wav"
            media_type = scene.get("media_type", "image")
            asset = {
                "scene_index": s_idx,
                "voiceover": scene["voiceover"],
                "prompt": scene.get("prompt", ""),
                "media_type": media_type,
                "zoom_effect": scene.get("zoom_effect", "zoom_in"),
                "focus_x": scene.get("focus_x", 0.5),
                "focus_y": scene.get("focus_y", 0.5),
                "time_fit_strategy": scene.get("time_fit_strategy", "auto"),
                "has_audio": os.path.exists(audio_path) and os.path.getsize(audio_path) > 0,
                "audio_url": f"/api/v6-scene-audio/{job_id}/{s_idx}",
            }
            if media_type == "image":
                image_path = f"{job_dir}/v6_scene_{s_idx}_image.jpg"
                asset["has_image"] = os.path.exists(image_path) and os.path.getsize(image_path) > 0
                asset["image_url"] = f"/api/v6-scene-image/{job_id}/{s_idx}"
                asset["has_video"] = False
                asset["video_url"] = ""
            else:
                video_path = f"{job_dir}/v6_scene_{s_idx}_video.mp4"
                asset["has_image"] = False
                asset["image_url"] = ""
                asset["has_video"] = os.path.exists(video_path) and os.path.getsize(video_path) > 0
                asset["video_url"] = f"/api/v6-scene-video/{job_id}/{s_idx}"
            assets.append(asset)
        total = len(request_data.get("v6_scenes", []))
        return {"job_id": job_id, "version": "v6", "total_scenes": total, "assets": assets}

    if version == "v5":
        assets = []
        for s_idx, scene in enumerate(request_data.get("v5_scenes", [])):
            audio_path = f"{job_dir}/v5_scene_{s_idx}_audio.wav"
            video_path = f"{job_dir}/v5_scene_{s_idx}_video.mp4"
            assets.append({
                "scene_index": s_idx,
                "voiceover": scene["voiceover"],
                "prompt": scene["prompt"],
                "media_type": scene.get("media_type", "video"),
                "time_fit_strategy": scene.get("time_fit_strategy", "auto"),
                "has_audio": os.path.exists(audio_path) and os.path.getsize(audio_path) > 0,
                "audio_url": f"/api/v5-scene-audio/{job_id}/{s_idx}",
                "has_video": os.path.exists(video_path) and os.path.getsize(video_path) > 0,
                "video_url": f"/api/v5-scene-video/{job_id}/{s_idx}",
            })
        total = len(request_data.get("v5_scenes", []))
        return {"job_id": job_id, "version": "v5", "total_scenes": total, "assets": assets}

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
    # Determine version: check in-memory job first, then infer from files on disk
    if job_id in jobs:
        version = jobs[job_id]["request"].get("version", "v1")
    else:
        # Try to detect version from which audio file exists on disk
        job_dir = f"{JOBS_DIR}/{job_id}"
        if os.path.exists(f"{job_dir}/v5_scene_{scene_index}_audio.wav"):
            version = "v5"
        elif os.path.exists(f"{job_dir}/v6_scene_{scene_index}_audio.wav"):
            version = "v6"
        elif os.path.exists(f"{job_dir}/v2_scene_{scene_index}_audio.wav"):
            version = "v2"
        else:
            version = "v1"
    if version == "v5":
        audio_path = f"{JOBS_DIR}/{job_id}/v5_scene_{scene_index}_audio.wav"
    elif version == "v6":
        audio_path = f"{JOBS_DIR}/{job_id}/v6_scene_{scene_index}_audio.wav"
    elif version in ("v2", "v3"):
        audio_path = f"{JOBS_DIR}/{job_id}/v2_scene_{scene_index}_audio.wav"
    else:
        audio_path = f"{JOBS_DIR}/{job_id}/scene_{scene_index}_audio.wav"
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(audio_path, media_type="audio/wav")


@app.post("/api/regenerate-image/{job_id}/{scene_index}")
async def regenerate_image(job_id: str, scene_index: int):
    """Regenerate a specific scene's image."""
    job = _ensure_job_in_memory(job_id)
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
    job = _ensure_job_in_memory(job_id)
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
    job = _ensure_job_in_memory(job_id)
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
    job = _ensure_job_in_memory(job_id)

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
    job = _ensure_job_in_memory(job_id)
    if job["status"] not in ("approved", "completed"):
        raise HTTPException(status_code=400, detail="Assets must be approved before video preparation")

    request = VideoRequest(**job["request"])

    # Reset progress for video assembly phase
    job["status"] = "queued"
    job["progress"] = 0
    job["message"] = "Preparing video..."
    job["output_path"] = None

    if request.version == "v5":
        background_tasks.add_task(run_v5_video_assembly, job_id, request)
    elif request.version == "v6":
        background_tasks.add_task(run_v6_video_assembly, job_id, request)
    elif request.version == "v3":
        background_tasks.add_task(run_v3_video_assembly, job_id, request)
    elif request.version == "v2":
        background_tasks.add_task(run_v2_video_assembly, job_id, request)
    else:
        background_tasks.add_task(run_video_assembly, job_id, request)
    return {"job_id": job_id}


class TimeFitUpdate(BaseModel):
    scene_index: int
    time_fit_strategy: str  # "auto", "trim", "cinematic_slow_mo", "loop_or_freeze"


@app.post("/api/update-time-fit/{job_id}")
async def update_time_fit(job_id: str, update: TimeFitUpdate):
    """Update the time_fit_strategy for a V5 or V6 video-clip scene so the user
    can re-render without regenerating audio or images."""
    job = _ensure_job_in_memory(job_id)
    request_data = job["request"]
    version = request_data.get("version", "v1")

    valid_strategies = {"auto", "trim", "cinematic_slow_mo", "loop_or_freeze"}
    if update.time_fit_strategy not in valid_strategies:
        raise HTTPException(status_code=400, detail=f"Invalid strategy. Must be one of: {valid_strategies}")

    if version == "v5":
        scenes = request_data.get("v5_scenes", [])
        if update.scene_index < 0 or update.scene_index >= len(scenes):
            raise HTTPException(status_code=400, detail="Invalid scene index")
        scenes[update.scene_index]["time_fit_strategy"] = update.time_fit_strategy
    elif version == "v6":
        scenes = request_data.get("v6_scenes", [])
        if update.scene_index < 0 or update.scene_index >= len(scenes):
            raise HTTPException(status_code=400, detail="Invalid scene index")
        if scenes[update.scene_index].get("media_type", "image") != "video":
            raise HTTPException(status_code=400, detail="Scene is not a video scene")
        scenes[update.scene_index]["time_fit_strategy"] = update.time_fit_strategy
    else:
        raise HTTPException(status_code=400, detail="time_fit_strategy only applies to V5 and V6 jobs")

    return {"success": True, "scene_index": update.scene_index, "time_fit_strategy": update.time_fit_strategy}


@app.get("/api/list-jobs")
async def list_jobs():
    """Return summary of all in-memory jobs (for My Videos history tab)."""
    summary = []
    for jid, job in jobs.items():
        req = job.get("request", {})
        summary.append({
            "id": jid,
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"],
            "version": req.get("version", "v1"),
            "has_output": bool(job.get("output_path") and os.path.exists(job["output_path"])),
        })
    return {"jobs": summary}


@app.get("/api/list-disk-jobs")
async def list_disk_jobs():
    """Scan the jobs directory on disk and return all jobs found (for resume/recovery)."""
    disk_jobs = []
    if not os.path.isdir(JOBS_DIR):
        return {"jobs": []}
    for job_id in os.listdir(JOBS_DIR):
        job_dir = f"{JOBS_DIR}/{job_id}"
        request_path = f"{job_dir}/request.json"
        if not os.path.isdir(job_dir) or not os.path.exists(request_path):
            continue
        try:
            with open(request_path) as f:
                request_data = json.load(f)
            output_path = f"{job_dir}/output.mp4"
            has_video = os.path.exists(output_path) and os.path.getsize(output_path) > 0
            mtime = os.path.getmtime(request_path)
            disk_jobs.append({
                "id": job_id,
                "version": request_data.get("version", "v1"),
                "has_video": has_video,
                "in_memory": job_id in jobs,
                "timestamp": int(mtime * 1000),
            })
        except Exception:
            continue
    disk_jobs.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"jobs": disk_jobs}


@app.post("/api/load-job/{job_id}")
async def load_job(job_id: str):
    """Load a job from disk into memory (for resuming after server restart)."""
    job_dir = f"{JOBS_DIR}/{job_id}"
    request_path = f"{job_dir}/request.json"
    if not os.path.isdir(job_dir) or not os.path.exists(request_path):
        raise HTTPException(status_code=404, detail="Job not found on disk")
    try:
        with open(request_path) as f:
            request_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read job metadata: {e}")

    status, output_path = _detect_disk_status(job_id, request_data)
    progress = 100 if status == "completed" else (80 if status == "assets_ready" else 0)
    message = {
        "completed": "Loaded from disk -- video ready.",
        "assets_ready": "Loaded from disk -- assets ready for review.",
        "failed": "Loaded from disk -- assets not found.",
    }.get(status, "Loaded from disk.")

    jobs[job_id] = {
        "id": job_id,
        "status": status,
        "progress": progress,
        "message": message,
        "output_path": output_path,
        "request": request_data,
    }
    return {"job_id": job_id, "status": status, "version": request_data.get("version", "v1")}


@app.get("/api/progress/{job_id}")
async def get_progress(job_id: str, request: Request):
    _ensure_job_in_memory(job_id)  # raises 404 if not found on disk either

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
    return _ensure_job_in_memory(job_id)


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    # Try in-memory job first
    if job_id in jobs:
        job = jobs[job_id]
        if job["status"] == "completed" and job.get("output_path") and os.path.exists(job["output_path"]):
            return FileResponse(job["output_path"], media_type="video/mp4", filename="generated_video.mp4")
    # Fallback: serve directly from disk
    output_path = f"{JOBS_DIR}/{job_id}/output.mp4"
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return FileResponse(output_path, media_type="video/mp4", filename="generated_video.mp4")
    raise HTTPException(status_code=404, detail="Video not found")


@app.post("/api/v5-upload-video/{job_id}/{scene_index}")
async def v5_upload_video(job_id: str, scene_index: int, file: UploadFile = File(...)):
    """Upload a video clip for a specific V5 scene."""
    job = _ensure_job_in_memory(job_id)
    version = job["request"].get("version", "v1")
    if version != "v5":
        raise HTTPException(status_code=400, detail="This endpoint is only for V5 jobs")

    v5_scenes = job["request"].get("v5_scenes", [])
    if scene_index < 0 or scene_index >= len(v5_scenes):
        raise HTTPException(status_code=400, detail="Invalid scene index")

    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are accepted")

    job_dir = f"{JOBS_DIR}/{job_id}"
    os.makedirs(job_dir, exist_ok=True)
    video_path = f"{job_dir}/v5_scene_{scene_index}_video.mp4"

    try:
        contents = await file.read()
        with open(video_path, "wb") as f:
            f.write(contents)
        return {
            "success": True,
            "message": f"Video uploaded for scene {scene_index + 1}",
            "video_url": f"/api/v5-scene-video/{job_id}/{scene_index}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v5-scene-video/{job_id}/{scene_index}")
async def get_v5_scene_video(job_id: str, scene_index: int):
    """Serve an uploaded V5 scene video clip."""
    video_path = f"{JOBS_DIR}/{job_id}/v5_scene_{scene_index}_video.mp4"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        video_path, media_type="video/mp4",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.get("/api/v5-scene-audio/{job_id}/{scene_index}")
async def get_v5_scene_audio(job_id: str, scene_index: int):
    """Serve the TTS audio for a V5 scene."""
    audio_path = f"{JOBS_DIR}/{job_id}/v5_scene_{scene_index}_audio.wav"
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(audio_path, media_type="audio/wav")


@app.get("/api/v5-dashboard-status/{job_id}")
async def v5_dashboard_status(job_id: str):
    """Check whether all V5 scenes have an uploaded video clip."""
    job = _ensure_job_in_memory(job_id)
    v5_scenes = job["request"].get("v5_scenes", [])
    job_dir = f"{JOBS_DIR}/{job_id}"

    scene_status = []
    all_ready = True
    for idx in range(len(v5_scenes)):
        has_video = (
            os.path.exists(f"{job_dir}/v5_scene_{idx}_video.mp4")
            and os.path.getsize(f"{job_dir}/v5_scene_{idx}_video.mp4") > 0
        )
        has_audio = (
            os.path.exists(f"{job_dir}/v5_scene_{idx}_audio.wav")
            and os.path.getsize(f"{job_dir}/v5_scene_{idx}_audio.wav") > 0
        )
        if not has_video:
            all_ready = False
        scene_status.append({
            "scene_index": idx,
            "has_video": has_video,
            "has_audio": has_audio,
        })

    return {"all_ready": all_ready, "scenes": scene_status}


# ── V6 asset endpoints ───────────────────────────────────────────────────────

@app.get("/api/v6-scene-image/{job_id}/{scene_index}")
async def get_v6_scene_image(job_id: str, scene_index: int):
    """Serve a generated image for a V6 image scene."""
    image_path = f"{JOBS_DIR}/{job_id}/v6_scene_{scene_index}_image.jpg"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        image_path, media_type="image/jpeg",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.get("/api/v6-scene-audio/{job_id}/{scene_index}")
async def get_v6_scene_audio(job_id: str, scene_index: int):
    """Serve the TTS audio for a V6 scene."""
    audio_path = f"{JOBS_DIR}/{job_id}/v6_scene_{scene_index}_audio.wav"
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(audio_path, media_type="audio/wav")


@app.get("/api/v6-scene-video/{job_id}/{scene_index}")
async def get_v6_scene_video(job_id: str, scene_index: int):
    """Serve an uploaded video clip for a V6 video scene."""
    video_path = f"{JOBS_DIR}/{job_id}/v6_scene_{scene_index}_video.mp4"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        video_path, media_type="video/mp4",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.post("/api/v6-upload-video/{job_id}/{scene_index}")
async def v6_upload_video(job_id: str, scene_index: int, file: UploadFile = File(...)):
    """Upload a video clip for a specific V6 video scene."""
    job = _ensure_job_in_memory(job_id)
    version = job["request"].get("version", "v1")
    if version != "v6":
        raise HTTPException(status_code=400, detail="This endpoint is only for V6 jobs")

    v6_scenes = job["request"].get("v6_scenes", [])
    if scene_index < 0 or scene_index >= len(v6_scenes):
        raise HTTPException(status_code=400, detail="Invalid scene index")

    scene = v6_scenes[scene_index]
    if scene.get("media_type", "image") != "video":
        raise HTTPException(status_code=400, detail="Scene is not a video scene")

    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are accepted")

    job_dir = f"{JOBS_DIR}/{job_id}"
    os.makedirs(job_dir, exist_ok=True)
    video_path = f"{job_dir}/v6_scene_{scene_index}_video.mp4"

    try:
        contents = await file.read()
        with open(video_path, "wb") as f:
            f.write(contents)
        return {
            "success": True,
            "message": f"Video uploaded for V6 scene {scene_index + 1}",
            "video_url": f"/api/v6-scene-video/{job_id}/{scene_index}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v6-upload-image/{job_id}/{scene_index}")
async def v6_upload_image(job_id: str, scene_index: int, file: UploadFile = File(...)):
    """Upload a custom image for a specific V6 image scene (replaces AI-generated image)."""
    job = _ensure_job_in_memory(job_id)
    version = job["request"].get("version", "v1")
    if version != "v6":
        raise HTTPException(status_code=400, detail="This endpoint is only for V6 jobs")

    v6_scenes = job["request"].get("v6_scenes", [])
    if scene_index < 0 or scene_index >= len(v6_scenes):
        raise HTTPException(status_code=400, detail="Invalid scene index")

    scene = v6_scenes[scene_index]
    if scene.get("media_type", "image") == "video":
        raise HTTPException(status_code=400, detail="Scene is not an image scene")

    allowed_extensions = (".jpg", ".jpeg", ".png", ".webp")
    fname = (file.filename or "").lower()
    if not any(fname.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, or WEBP images are accepted")

    job_dir = f"{JOBS_DIR}/{job_id}"
    os.makedirs(job_dir, exist_ok=True)
    image_path = f"{job_dir}/v6_scene_{scene_index}_image.jpg"

    try:
        contents = await file.read()
        # Convert any accepted format to JPEG using PIL
        img = PILImage.open(io.BytesIO(contents)).convert("RGB")
        img.save(image_path, "JPEG", quality=92)
        return {
            "success": True,
            "message": f"Image uploaded for V6 scene {scene_index + 1}",
            "image_url": f"/api/v6-scene-image/{job_id}/{scene_index}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class V6FocusPointUpdate(BaseModel):
    scene_index: int
    focus_x: float
    focus_y: float
    zoom_effect: Optional[str] = None


@app.post("/api/v6-update-scene/{job_id}")
async def v6_update_scene(job_id: str, update: V6FocusPointUpdate):
    """Save focus point and optional zoom effect for a V6 image scene."""
    job = _ensure_job_in_memory(job_id)

    v6_scenes = job["request"].get("v6_scenes", [])
    if update.scene_index < 0 or update.scene_index >= len(v6_scenes):
        raise HTTPException(status_code=400, detail="Invalid scene index")

    fx = max(0.0, min(1.0, update.focus_x))
    fy = max(0.0, min(1.0, update.focus_y))
    v6_scenes[update.scene_index]["focus_x"] = fx
    v6_scenes[update.scene_index]["focus_y"] = fy
    if update.zoom_effect is not None:
        v6_scenes[update.scene_index]["zoom_effect"] = update.zoom_effect
    return {
        "success": True,
        "focus_x": fx,
        "focus_y": fy,
        "zoom_effect": v6_scenes[update.scene_index].get("zoom_effect"),
    }


@app.post("/api/v6-regenerate-image/{job_id}/{scene_index}")
async def v6_regenerate_image(job_id: str, scene_index: int):
    """Regenerate the AI image for a specific V6 image scene."""
    job = _ensure_job_in_memory(job_id)
    if job["status"] != "assets_ready":
        raise HTTPException(status_code=400, detail="Assets not ready for regeneration")

    request_data = job["request"]
    v6_scenes = request_data.get("v6_scenes", [])
    if scene_index < 0 or scene_index >= len(v6_scenes):
        raise HTTPException(status_code=400, detail="Invalid scene index")
    scene = v6_scenes[scene_index]
    if scene.get("media_type", "image") != "image":
        raise HTTPException(status_code=400, detail="Scene is not an image scene")

    image_path = f"{JOBS_DIR}/{job_id}/v6_scene_{scene_index}_image.jpg"
    if os.path.exists(image_path):
        os.remove(image_path)

    try:
        _generate_image_for_provider(
            request_data.get("image_provider", "gemini"),
            scene["prompt"],
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
            return {"success": True, "message": f"V6 image for scene {scene_index + 1} regenerated"}
        raise HTTPException(status_code=500, detail="Image regeneration failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/test-audio")
async def test_audio(request: TestAudioRequest):
    test_id = str(uuid.uuid4())
    test_dir = f"{JOBS_DIR}/test_{test_id}"
    os.makedirs(test_dir, exist_ok=True)
    audio_path = f"{test_dir}/test_audio.wav"

    try:
        success = _generate_audio_for_provider(
            request.speech_provider, request.text, audio_path,
            speech_model=request.speech_model,
            speech_voice=request.speech_voice,
            gemini_api_key=request.gemini_api_key or "",
            openai_api_key=request.openai_api_key or "",
            elevenlabs_api_key=request.elevenlabs_api_key or "",
        )

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
            {"value": "elevenlabs", "label": "ElevenLabs"},
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
            "elevenlabs": [
                {"value": "eleven_multilingual_v2", "label": "Multilingual v2"},
                {"value": "eleven_turbo_v2_5", "label": "Turbo v2.5"},
                {"value": "eleven_flash_v2_5", "label": "Flash v2.5"},
                {"value": "eleven_turbo_v2", "label": "Turbo v2"},
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
            "elevenlabs": [
                {"value": "3TStB8f3X3To0Uj5R7RK", "label": "Samuel (Male, Natural) -- Default"},
                {"value": "21m00Tcm4TlvDq8ikWAM", "label": "Rachel (Female, Calm)"},
                {"value": "AZnzlk1XvdvUeBnXmlld", "label": "Domi (Female, Strong)"},
                {"value": "EXAVITQu4vr4xnSDxMaL", "label": "Bella (Female, Soft)"},
                {"value": "ErXwobaYiN019PkySvjV", "label": "Antoni (Male, Warm)"},
                {"value": "MF3mGyEYCl7XYWbV9V6O", "label": "Elli (Female, Emotional)"},
                {"value": "TxGEqnHWrfWFTfGW9XjX", "label": "Josh (Male, Deep)"},
                {"value": "VR6AewLTigWG4xSOukaG", "label": "Arnold (Male, Crisp)"},
                {"value": "pNInz6obpgDQGcFmaJgB", "label": "Adam (Male, Deep)"},
                {"value": "yoZ06aMxZJJ28mfd3POQ", "label": "Sam (Male, Raspy)"},
                {"value": "XB0fDUnXU5powFXDhCwa", "label": "Charlotte (Female, Confident)"},
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
