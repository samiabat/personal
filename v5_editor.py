"""
V5 Video Editor – Smart Time-Remapping Engine
==============================================
Handles AI-generated 6-second video clips (e.g. from Grok) and fits them to
variable-length TTS voiceover audio using three strategies:

  * **trim**              – audio < 6 s → trim video to match.
  * **cinematic_slow_mo** – 6 s < audio ≤ 12 s → time-stretch the clip.
  * **loop_or_freeze**    – audio > 12 s → loop/ping-pong + freeze-frame.

All uploaded clips have their original audio **stripped** so only the
voiceover track is present in the final output.
"""

from __future__ import annotations

import os
from typing import Optional

from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx import TimeMirror

# ── Constants ────────────────────────────────────────────────────────────────
GROK_CLIP_DURATION = 6.0          # seconds – canonical Grok clip length
SLOW_MO_CEILING    = 12.0         # above this → loop / freeze instead of slow-mo
FPS                = 24


# ── Time-Fit helpers ─────────────────────────────────────────────────────────

def _trim_clip(clip: VideoFileClip, target_dur: float) -> VideoFileClip:
    """Trim the clip to *target_dur* keeping the centre portion."""
    clip_dur = clip.duration
    if target_dur >= clip_dur:
        return clip
    start = (clip_dur - target_dur) / 2.0
    return clip.subclipped(start, start + target_dur)


def _slow_mo_clip(clip: VideoFileClip, target_dur: float) -> VideoFileClip:
    """Time-stretch *clip* so it lasts exactly *target_dur* seconds.

    Uses MoviePy's ``with_speed_scaled`` to slow the playback.
    """
    if clip.duration <= 0:
        return clip
    speed_factor = clip.duration / target_dur        # < 1.0 → slower
    return clip.with_speed_scaled(speed_factor)


def _reverse_clip(clip: VideoFileClip) -> VideoFileClip:
    """Return a time-reversed copy of *clip* (plays backward)."""
    return clip.with_effects([TimeMirror()])


def _loop_and_freeze_clip(clip: VideoFileClip, target_dur: float) -> VideoFileClip:
    """Fill *target_dur* by looping the clip with alternating direction
    (ping-pong) and freezing the last frame for any remaining time.
    """
    clip_dur = clip.duration
    if clip_dur <= 0:
        return clip

    segments: list[VideoFileClip] = []
    remaining = target_dur
    forward = True

    while remaining > 0:
        if remaining >= clip_dur:
            seg = clip.copy() if forward else _reverse_clip(clip)
            segments.append(seg)
            remaining -= clip_dur
        else:
            # Partial segment – slow-mo the remainder so it still looks smooth
            seg = clip.copy() if forward else _reverse_clip(clip)
            speed = clip_dur / remaining
            seg = seg.with_speed_scaled(speed)
            segments.append(seg)
            remaining = 0
        forward = not forward

    result = concatenate_videoclips(segments)
    # Safety clamp to exact duration
    if result.duration > target_dur:
        result = result.subclipped(0, target_dur)
    return result


def fit_clip_to_audio(
    clip: VideoFileClip,
    audio_duration: float,
    strategy: str = "auto",
) -> VideoFileClip:
    """Apply the correct time-fit strategy to make *clip* match *audio_duration*.

    Parameters
    ----------
    strategy : str
        ``"auto"``               – choose automatically based on ratio.
        ``"trim"``               – always trim.
        ``"cinematic_slow_mo"``  – always slow-mo.
        ``"loop_or_freeze"``     – always loop / ping-pong.
    """
    clip_dur = clip.duration

    if strategy == "auto":
        if audio_duration <= clip_dur:
            strategy = "trim"
        elif audio_duration <= SLOW_MO_CEILING:
            strategy = "cinematic_slow_mo"
        else:
            strategy = "loop_or_freeze"

    if strategy == "trim":
        return _trim_clip(clip, audio_duration)
    elif strategy == "cinematic_slow_mo":
        return _slow_mo_clip(clip, audio_duration)
    elif strategy == "loop_or_freeze":
        return _loop_and_freeze_clip(clip, audio_duration)
    else:
        # Fallback: auto
        return fit_clip_to_audio(clip, audio_duration, strategy="auto")


# ── Main assembly ────────────────────────────────────────────────────────────

def assemble_v5_video(
    scenes: list[dict],
    job_dir: str,
    output_filename: str = "output.mp4",
    resolution: str = "1080p",
    orientation: str = "landscape",
    enable_subtitles: bool = False,
    subtitle_style: str = "cinematic",
) -> str:
    """Assemble a V5 video from per-scene video clips + TTS audio.

    Parameters
    ----------
    scenes : list[dict]
        Each dict must contain:
        - ``voiceover``          (str)  – voiceover text (for reference).
        - ``prompt``             (str)  – visual prompt (for reference).
        - ``time_fit_strategy``  (str)  – "auto" | "trim" | "cinematic_slow_mo" | "loop_or_freeze".
        File paths are resolved by convention inside *job_dir*:
        - ``v5_scene_{idx}_audio.wav``
        - ``v5_scene_{idx}_video.mp4``
    job_dir : str
        Directory containing all per-scene assets.
    output_filename : str
        Name of the final MP4 inside *job_dir*.
    resolution, orientation
        Target resolution / orientation (used for final resize).

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

    final_clips: list[VideoFileClip] = []

    for idx, scene in enumerate(scenes):
        video_path = os.path.join(job_dir, f"v5_scene_{idx}_video.mp4")
        audio_path = os.path.join(job_dir, f"v5_scene_{idx}_audio.wav")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video clip missing for scene {idx}: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file missing for scene {idx}: {audio_path}")

        # Load & strip original audio from uploaded clip
        raw_clip = VideoFileClip(video_path).without_audio()

        # Load TTS audio to get its duration
        tts_audio = AudioFileClip(audio_path)
        audio_dur = tts_audio.duration

        # Time-fit the video clip
        strategy = scene.get("time_fit_strategy", "auto")
        fitted = fit_clip_to_audio(raw_clip, audio_dur, strategy=strategy)

        # Resize to target resolution
        fitted = fitted.resized((target_w, target_h))

        # Attach the TTS audio
        fitted = fitted.with_audio(tts_audio)

        final_clips.append(fitted)

    # Concatenate all scene clips
    if len(final_clips) == 1:
        final_video = final_clips[0]
    else:
        final_video = concatenate_videoclips(final_clips, method="compose")

    # Optionally burn in subtitles
    if enable_subtitles:
        try:
            from subtitle_renderer import render_subtitles
            final_video = render_subtitles(
                final_video,
                scenes,
                style=subtitle_style,
            )
        except Exception:
            pass  # graceful degradation – subtitles are optional

    output_path = os.path.join(job_dir, output_filename)
    final_video.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
    )
    return output_path
