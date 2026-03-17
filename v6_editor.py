"""
V6 Video Editor – Hybrid Image + Video Scene Engine
====================================================
V6 allows each scene to be either an **image scene** or a **video scene**:

* **Image scene** (``media_type="image"``)
    – An AI-generated image is animated with a configurable zoom/pan effect
      (``zoom_effect``) centred on a director-chosen focus point
      (``focus_x``, ``focus_y``).  The clip length exactly matches the TTS
      voiceover.  Available effects:

      * ``none``        – static frame for the full duration.
      * ``zoom_in``     – cinematic ease-in zoom toward the focus point.
      * ``zoom_out``    – cinematic ease-out zoom away from the focus point.
      * ``ken_burns``   – gentle pan + zoom across the image (ignores focus).

* **Video scene** (``media_type="video"``)
    – A user-supplied video clip is time-fitted to the voiceover duration
      using the same strategy as V5 (auto / trim / cinematic_slow_mo /
      loop_or_freeze).  The clip's original audio is stripped; only the TTS
      voiceover is used.

Previous versions (V1–V5) are **not affected** by this module.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image
from moviepy import (
    AudioFileClip,
    VideoClip,
    VideoFileClip,
    concatenate_videoclips,
)

# Re-use V5 time-fit helpers so we never duplicate that logic
from v5_editor import fit_clip_to_audio

# Re-use V2/V3 image-prep utility
from v2_editor import _prepare_image, _smoothstep

# ── Constants ────────────────────────────────────────────────────────────────
FPS = 24
ZOOM_SCALE = 1.18          # Peak zoom factor for zoom_in / zoom_out effects
KEN_BURNS_SCALE = 1.12     # Peak scale for ken_burns effect
KEN_BURNS_PAN_FRAC = 0.06  # Fraction of image width/height to pan


# ── Image-scene effect builders ──────────────────────────────────────────────

def _static_image_clip(image_path: str, duration: float,
                        target_size: tuple, fps: int = FPS) -> VideoClip:
    """Return a static (no-motion) clip from *image_path*."""
    img_arr, new_w, new_h = _prepare_image(image_path, target_size, 1.0)
    w, h = target_size
    frame = np.array(Image.fromarray(img_arr).resize((w, h), Image.LANCZOS))

    def make_frame(_t):
        return frame

    return VideoClip(make_frame, duration=duration).with_fps(fps)


def _zoom_in_clip(image_path: str, duration: float, target_size: tuple,
                  focus_x: float = 0.5, focus_y: float = 0.5,
                  fps: int = FPS) -> VideoClip:
    """Cinematic ease-in zoom toward *focus_x*, *focus_y* (normalised 0–1)."""
    img_arr, new_w, new_h = _prepare_image(image_path, target_size, ZOOM_SCALE)
    w, h = target_size
    fcx = int(np.clip(focus_x, 0.0, 1.0) * new_w)
    fcy = int(np.clip(focus_y, 0.0, 1.0) * new_h)

    ease_dur = min(duration * 0.35, 1.2)   # ease-in and ease-out phase duration (each)
    hold_dur = max(0.0, duration - ease_dur * 2)
    ease_out_dur = ease_dur

    def make_frame(t):
        if t < ease_dur:
            progress = _smoothstep(t / max(ease_dur, 1e-6))
            zoom = 1.0 + (ZOOM_SCALE - 1.0) * progress
        elif t < ease_dur + hold_dur:
            zoom = ZOOM_SCALE
        else:
            eo_t = t - ease_dur - hold_dur
            progress = _smoothstep(eo_t / max(ease_out_dur, 1e-6))
            zoom = ZOOM_SCALE + (1.0 - ZOOM_SCALE) * progress

        zoom = max(zoom, 1.0)
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)
        x1 = max(0, min(fcx - crop_w // 2, new_w - crop_w))
        y1 = max(0, min(fcy - crop_h // 2, new_h - crop_h))
        cropped = img_arr[y1:y1 + crop_h, x1:x1 + crop_w]
        return np.array(Image.fromarray(cropped).resize((w, h), Image.LANCZOS))

    return VideoClip(make_frame, duration=duration).with_fps(fps)


def _zoom_out_clip(image_path: str, duration: float, target_size: tuple,
                   focus_x: float = 0.5, focus_y: float = 0.5,
                   fps: int = FPS) -> VideoClip:
    """Cinematic ease-out zoom, starting at *ZOOM_SCALE* then pulling back."""
    img_arr, new_w, new_h = _prepare_image(image_path, target_size, ZOOM_SCALE)
    w, h = target_size
    fcx = int(np.clip(focus_x, 0.0, 1.0) * new_w)
    fcy = int(np.clip(focus_y, 0.0, 1.0) * new_h)

    hold_dur = min(duration * 0.15, 0.6)   # brief hold at peak
    ease_out_dur = min(duration * 0.45, 1.5)
    ease_in_dur = max(0.0, duration - hold_dur - ease_out_dur)

    def make_frame(t):
        if t < ease_in_dur:
            zoom = ZOOM_SCALE
        elif t < ease_in_dur + hold_dur:
            zoom = ZOOM_SCALE
        else:
            eo_t = t - ease_in_dur - hold_dur
            progress = _smoothstep(eo_t / max(ease_out_dur, 1e-6))
            zoom = ZOOM_SCALE + (1.0 - ZOOM_SCALE) * progress

        zoom = max(zoom, 1.0)
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)
        x1 = max(0, min(fcx - crop_w // 2, new_w - crop_w))
        y1 = max(0, min(fcy - crop_h // 2, new_h - crop_h))
        cropped = img_arr[y1:y1 + crop_h, x1:x1 + crop_w]
        return np.array(Image.fromarray(cropped).resize((w, h), Image.LANCZOS))

    return VideoClip(make_frame, duration=duration).with_fps(fps)


def _ken_burns_clip(image_path: str, duration: float, target_size: tuple,
                    fps: int = FPS) -> VideoClip:
    """Gentle Ken Burns pan-and-zoom covering the full *duration*."""
    img_arr, new_w, new_h = _prepare_image(image_path, target_size, KEN_BURNS_SCALE)
    w, h = target_size

    # Pan from top-left crop to bottom-right crop while slowly zooming in
    pan_x = int(new_w * KEN_BURNS_PAN_FRAC)
    pan_y = int(new_h * KEN_BURNS_PAN_FRAC)

    def make_frame(t):
        progress = _smoothstep(t / max(duration, 1e-6))
        zoom = 1.0 + (KEN_BURNS_SCALE - 1.0) * progress * 0.5
        zoom = max(zoom, 1.0)
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)

        x_offset = int(pan_x * progress)
        y_offset = int(pan_y * progress)
        x1 = max(0, min(x_offset, new_w - crop_w))
        y1 = max(0, min(y_offset, new_h - crop_h))
        cropped = img_arr[y1:y1 + crop_h, x1:x1 + crop_w]
        return np.array(Image.fromarray(cropped).resize((w, h), Image.LANCZOS))

    return VideoClip(make_frame, duration=duration).with_fps(fps)


def _build_image_scene_clip(
    image_path: str,
    audio_duration: float,
    target_size: tuple,
    zoom_effect: str = "zoom_in",
    focus_x: float = 0.5,
    focus_y: float = 0.5,
    fps: int = FPS,
) -> VideoClip:
    """Create an animated still-image clip matching *audio_duration*.

    Parameters
    ----------
    zoom_effect : str
        ``"none"`` | ``"zoom_in"`` | ``"zoom_out"`` | ``"ken_burns"``
    focus_x, focus_y : float
        Normalised focus coordinates (0–1).  Used for zoom_in / zoom_out.
    """
    if zoom_effect == "zoom_in":
        return _zoom_in_clip(image_path, audio_duration, target_size,
                             focus_x=focus_x, focus_y=focus_y, fps=fps)
    elif zoom_effect == "zoom_out":
        return _zoom_out_clip(image_path, audio_duration, target_size,
                              focus_x=focus_x, focus_y=focus_y, fps=fps)
    elif zoom_effect == "ken_burns":
        return _ken_burns_clip(image_path, audio_duration, target_size, fps=fps)
    else:
        return _static_image_clip(image_path, audio_duration, target_size, fps=fps)


# ── Main assembly ────────────────────────────────────────────────────────────

def assemble_v6_video(
    scenes: list[dict],
    job_dir: str,
    output_filename: str = "output.mp4",
    resolution: str = "1080p",
    orientation: str = "landscape",
    enable_subtitles: bool = False,
    subtitle_style: str = "cinematic",
) -> str:
    """Assemble a V6 video from mixed image + video scenes.

    Parameters
    ----------
    scenes : list[dict]
        Each dict must contain:

        Common keys
        ~~~~~~~~~~~
        - ``voiceover``    (str)  – voiceover text (for reference).
        - ``media_type``   (str)  – ``"image"`` or ``"video"``.

        Image-scene keys
        ~~~~~~~~~~~~~~~~
        - ``prompt``       (str)  – image prompt (for reference).
        - ``zoom_effect``  (str)  – ``"none"`` | ``"zoom_in"`` (default) |
                                    ``"zoom_out"`` | ``"ken_burns"``.
        - ``focus_x``      (float) – zoom focus X (0–1, default 0.5).
        - ``focus_y``      (float) – zoom focus Y (0–1, default 0.5).

        Video-scene keys
        ~~~~~~~~~~~~~~~~
        - ``time_fit_strategy`` (str) – ``"auto"`` | ``"trim"`` |
                                        ``"cinematic_slow_mo"`` |
                                        ``"loop_or_freeze"``.

        Asset file naming (resolved inside *job_dir*):
        - ``v6_scene_{idx}_audio.wav``
        - ``v6_scene_{idx}_image.jpg``   (image scenes)
        - ``v6_scene_{idx}_video.mp4``   (video scenes)

    Returns
    -------
    str – absolute path to the rendered output file.
    """
    res_map = {
        "480p":  (854, 480),
        "720p":  (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4K":    (3840, 2160),
    }
    target_w, target_h = res_map.get(resolution, (1920, 1080))
    if orientation == "portrait":
        target_w, target_h = target_h, target_w
    target_size = (target_w, target_h)

    final_clips: list = []

    for idx, scene in enumerate(scenes):
        audio_path = os.path.join(job_dir, f"v6_scene_{idx}_audio.wav")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio missing for V6 scene {idx}: {audio_path}")

        tts_audio = AudioFileClip(audio_path)
        audio_dur = tts_audio.duration

        media_type = scene.get("media_type", "image")

        if media_type == "video":
            video_path = os.path.join(job_dir, f"v6_scene_{idx}_video.mp4")
            if not os.path.exists(video_path):
                raise FileNotFoundError(
                    f"Video clip missing for V6 scene {idx}: {video_path}"
                )
            raw_clip = VideoFileClip(video_path).without_audio()
            strategy = scene.get("time_fit_strategy", "auto")
            fitted = fit_clip_to_audio(raw_clip, audio_dur, strategy=strategy)
            fitted = fitted.resized(target_size)
        else:
            # image scene
            image_path = os.path.join(job_dir, f"v6_scene_{idx}_image.jpg")
            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"Image missing for V6 scene {idx}: {image_path}"
                )
            zoom_effect = scene.get("zoom_effect", "zoom_in")
            focus_x = float(scene.get("focus_x", 0.5))
            focus_y = float(scene.get("focus_y", 0.5))
            fitted = _build_image_scene_clip(
                image_path, audio_dur, target_size,
                zoom_effect=zoom_effect,
                focus_x=focus_x,
                focus_y=focus_y,
            )

        fitted = fitted.with_audio(tts_audio)
        final_clips.append(fitted)

    if len(final_clips) == 1:
        final_video = final_clips[0]
    else:
        final_video = concatenate_videoclips(final_clips, method="compose")

    if enable_subtitles:
        try:
            from subtitle_renderer import render_subtitles
            final_video = render_subtitles(
                final_video,
                scenes,
                style=subtitle_style,
            )
        except Exception as sub_err:
            import warnings
            warnings.warn(f"Subtitle rendering failed and was skipped: {sub_err}")

    output_path = os.path.join(job_dir, output_filename)
    final_video.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
    )
    return output_path
