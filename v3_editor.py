"""V3 Video Editing Engine.

Extends the V2 pipeline with *focus-point-aware* zoom and pop-scale effects.
When ``focus_x`` / ``focus_y`` are present on a visual beat the camera
centres on that point instead of the default image centre.

Zoom uses **constant-velocity** scaling with a Zoom → Hold → Reset sequence:

* **ZOOM_SPEED** – scale units gained per second (default 0.02).
* **HOLD_BREATH** – seconds to hold the peak scale after the zoom phase (1.0 s).
* **EASE_OUT_RESET** – seconds to smoothly interpolate back to 1.0 (0.8 s).

Additive zooming: consecutive ``zoom_in_slow`` beats continue from the
accumulated scale instead of resetting to 1.0.
"""

import numpy as np
from PIL import Image
from moviepy import AudioFileClip, VideoClip, concatenate_videoclips

# Re-use all V2 helpers that don't need focus-point changes
from v2_editor import (
    get_word_timestamps,
    find_trigger_timestamp,
    find_nearest_audio_peak,
    _smoothstep,
    _prepare_image,
    _create_static_clip,
    _create_shake_clip,
    apply_color_shift,
)

# ---------------------------------------------------------------------------
# Constant-velocity zoom constants
# ---------------------------------------------------------------------------

ZOOM_SPEED = 0.02         # scale units per second
HOLD_BREATH = 1.0         # seconds to hold peak scale after zoom phase
EASE_OUT_RESET = 0.8      # seconds to ease back to 1.0


# ---------------------------------------------------------------------------
# Focus-aware effect helpers
# ---------------------------------------------------------------------------

def _focus_center(focus_x: float, focus_y: float, new_w: int, new_h: int) -> tuple[int, int]:
    """Convert normalised focus floats (0–1) to pixel coordinates."""
    return (
        int(np.clip(focus_x, 0.0, 1.0) * new_w),
        int(np.clip(focus_y, 0.0, 1.0) * new_h),
    )


def _create_constant_velocity_zoom_clip(
    image_path: str,
    duration: float,
    target_size: tuple,
    start_scale: float = 1.0,
    zoom_speed: float = ZOOM_SPEED,
    hold_breath: float = HOLD_BREATH,
    ease_out_dur: float = EASE_OUT_RESET,
    include_reset: bool = True,
    focus_x: float = 0.5,
    focus_y: float = 0.5,
    fps: int = 24,
) -> VideoClip:
    """Constant-velocity zoom with Zoom → Hold → Reset phases.

    Parameters
    ----------
    start_scale : float
        Initial zoom scale (1.0 for fresh zoom, higher for additive).
    zoom_speed : float
        Scale units gained per second during the zoom phase.
    hold_breath : float
        Seconds to hold the peak scale after zooming.
    ease_out_dur : float
        Seconds to ease back to 1.0 after the hold.
    include_reset : bool
        When *False* the clip is pure zoom for the full *duration* (used for
        additive zooming when the next beat continues zooming).
    focus_x, focus_y : float
        Normalised focus coordinates (0–1).
    """
    img_arr, new_w, new_h = _prepare_image(image_path, target_size, 1.20)
    w, h = target_size
    fcx, fcy = _focus_center(focus_x, focus_y, new_w, new_h)

    if include_reset:
        # Allocate phases within the available duration
        zoom_phase_dur = max(0.0, duration - hold_breath - ease_out_dur)
        hold_phase_dur = min(hold_breath, max(0.0, duration - zoom_phase_dur))
        ease_phase_dur = max(0.0, duration - zoom_phase_dur - hold_phase_dur)
        peak_scale = start_scale + zoom_speed * zoom_phase_dur
    else:
        # Pure zoom – entire duration is the zoom phase
        zoom_phase_dur = duration
        hold_phase_dur = 0.0
        ease_phase_dur = 0.0
        peak_scale = start_scale + zoom_speed * zoom_phase_dur

    def make_frame(t):
        if t < zoom_phase_dur:
            # Phase 1 – constant-velocity zoom
            zoom = start_scale + zoom_speed * t
        elif t < zoom_phase_dur + hold_phase_dur:
            # Phase 2 – hold peak
            zoom = peak_scale
        else:
            # Phase 3 – smooth ease-out back to 1.0
            eo_t = t - zoom_phase_dur - hold_phase_dur
            progress = _smoothstep(eo_t / max(ease_phase_dur, 1e-6))
            zoom = peak_scale + (1.0 - peak_scale) * progress

        # Clamp zoom to prevent inversion
        zoom = max(zoom, 1.0)

        crop_w = int(w / zoom)
        crop_h = int(h / zoom)

        x1 = max(0, min(fcx - crop_w // 2, new_w - crop_w))
        y1 = max(0, min(fcy - crop_h // 2, new_h - crop_h))

        cropped = img_arr[y1:y1 + crop_h, x1:x1 + crop_w]
        return np.array(Image.fromarray(cropped).resize((w, h), Image.LANCZOS))

    return VideoClip(make_frame, duration=duration).with_fps(fps)


def _create_zoom_clip_focused(image_path: str, duration: float,
                               target_size: tuple, direction: str = "in",
                               fps: int = 24, hold_duration: float | None = None,
                               ease_duration: float = 0.5,
                               zoom_target: float = 1.15,
                               focus_x: float = 0.5,
                               focus_y: float = 0.5) -> VideoClip:
    """Cinematic zoom with optional focus point (used for zoom_out_slow).

    Parameters ``focus_x`` and ``focus_y`` are normalised floats in [0, 1].
    ``(0.5, 0.5)`` is the image centre (V2 default behaviour).
    """
    img_arr, new_w, new_h = _prepare_image(image_path, target_size, 1.20)
    w, h = target_size

    max_phase_dur = duration / 3.0
    ease_in_dur = min(ease_duration, max_phase_dur)
    ease_out_dur = min(ease_duration, max_phase_dur)

    if hold_duration is not None:
        hold_dur = hold_duration
        total = ease_in_dur + hold_dur + ease_out_dur
        if total > duration:
            hold_dur = max(0.0, duration - ease_in_dur - ease_out_dur)
    else:
        hold_dur = max(0.0, duration - ease_in_dur - ease_out_dur)

    if direction == "in":
        zoom_base = 1.0
        zoom_peak = zoom_target
    else:
        zoom_base = zoom_target
        zoom_peak = 1.0

    # Focus centre in pixel space
    fcx, fcy = _focus_center(focus_x, focus_y, new_w, new_h)

    def make_frame(t):
        if t < ease_in_dur:
            progress = _smoothstep(t / max(ease_in_dur, 1e-6))
            zoom = zoom_base + (zoom_peak - zoom_base) * progress
        elif t < ease_in_dur + hold_dur:
            zoom = zoom_peak
        else:
            ease_out_t = t - ease_in_dur - hold_dur
            progress = _smoothstep(ease_out_t / max(ease_out_dur, 1e-6))
            zoom = zoom_peak + (zoom_base - zoom_peak) * progress

        crop_w = int(w / zoom)
        crop_h = int(h / zoom)

        x1 = max(0, min(fcx - crop_w // 2, new_w - crop_w))
        y1 = max(0, min(fcy - crop_h // 2, new_h - crop_h))

        cropped = img_arr[y1:y1 + crop_h, x1:x1 + crop_w]
        return np.array(Image.fromarray(cropped).resize((w, h), Image.LANCZOS))

    return VideoClip(make_frame, duration=duration).with_fps(fps)


def _create_pop_scale_clip_focused(image_path: str, duration: float,
                                    target_size: tuple, fps: int = 24,
                                    focus_x: float = 0.5,
                                    focus_y: float = 0.5) -> VideoClip:
    """Quick zoom-pop centred on the focus point."""
    img_arr, new_w, new_h = _prepare_image(image_path, target_size, 1.20)
    w, h = target_size
    pop_dur = min(0.3, duration * 0.5)

    fcx, fcy = _focus_center(focus_x, focus_y, new_w, new_h)

    def make_frame(t):
        if t < pop_dur:
            progress = t / pop_dur
            zoom = 1.0 + 0.08 * np.sin(progress * np.pi)
        else:
            zoom = 1.0

        crop_w = int(w / zoom)
        crop_h = int(h / zoom)

        x1 = max(0, min(fcx - crop_w // 2, new_w - crop_w))
        y1 = max(0, min(fcy - crop_h // 2, new_h - crop_h))

        cropped = img_arr[y1:y1 + crop_h, x1:x1 + crop_w]
        return np.array(Image.fromarray(cropped).resize((w, h), Image.LANCZOS))

    return VideoClip(make_frame, duration=duration).with_fps(fps)


# ---------------------------------------------------------------------------
# V3 assembly
# ---------------------------------------------------------------------------

ZOOM_EFFECTS = {"zoom_in_slow", "zoom_out_slow", "pop_scale"}


def assemble_v3_video(image_paths: list[str], audio_path: str,
                      visual_beats: list[dict], word_timestamps: list[dict],
                      output_path: str, target_size: tuple = (1920, 1080),
                      fps: int = 24) -> str:
    """Build the final V3 video with constant-velocity zoom and additive
    zooming support.

    Zoom behaviour
    --------------
    * ``zoom_in_slow`` uses a **constant velocity** (``ZOOM_SPEED``) instead
      of scaling to a fixed target over the segment duration.
    * Consecutive ``zoom_in_slow`` beats are **additive**: each one starts
      from the scale the previous one reached.
    * After the last ``zoom_in_slow`` in a run the clip includes a
      **Hold** (``HOLD_BREATH``) then a smooth **Ease-Out** back to 1.0
      (``EASE_OUT_RESET``).
    * If the next beat is a ``hard_cut`` the scale resets instantly (no
      hold / ease-out).

    The *visual_beats* dicts may contain optional ``focus_x`` and ``focus_y``
    float keys (0.0–1.0).  When absent the image centre (0.5, 0.5) is used.
    """
    audio_clip = AudioFileClip(audio_path)
    total_duration = audio_clip.duration

    # Resolve beat timestamps
    beat_events = []
    for beat in visual_beats:
        t = find_trigger_timestamp(word_timestamps, beat["trigger_word"])
        beat_events.append({
            "time": t,
            "effect": beat["effect"],
            "image_index": beat["image_index"],
            "color_grade": beat.get("color_grade"),
            "focus_x": beat.get("focus_x", 0.5),
            "focus_y": beat.get("focus_y", 0.5),
        })

    beat_events.sort(key=lambda x: x["time"])

    for ev in beat_events:
        if ev["effect"] == "audio_reactive_shake":
            ev["peak_time"] = find_nearest_audio_peak(audio_path, ev["time"])

    # Build clip segments
    segments: list[VideoClip] = []

    if beat_events and beat_events[0]["time"] > 0:
        pre_img = image_paths[beat_events[0]["image_index"]]
        segments.append(_create_static_clip(pre_img, beat_events[0]["time"], target_size, fps))

    # Track accumulated scale for additive zoom_in_slow sequences
    accumulated_scale = 1.0

    for i, ev in enumerate(beat_events):
        start = ev["time"]
        end = beat_events[i + 1]["time"] if i + 1 < len(beat_events) else total_duration

        if end <= start:
            continue

        seg_duration = end - start
        img_path = image_paths[ev["image_index"]]
        effect = ev["effect"]
        fx = ev["focus_x"]
        fy = ev["focus_y"]

        if effect == "zoom_in_slow":
            next_effect = beat_events[i + 1]["effect"] if i + 1 < len(beat_events) else None

            if next_effect == "zoom_in_slow":
                # Additive: pure zoom for the full segment, no hold/reset
                clip = _create_constant_velocity_zoom_clip(
                    img_path, seg_duration, target_size,
                    start_scale=accumulated_scale,
                    include_reset=False,
                    focus_x=fx, focus_y=fy, fps=fps,
                )
                accumulated_scale += ZOOM_SPEED * seg_duration
            elif next_effect == "hard_cut":
                # Zoom the full segment then instant reset (no hold/ease-out)
                clip = _create_constant_velocity_zoom_clip(
                    img_path, seg_duration, target_size,
                    start_scale=accumulated_scale,
                    include_reset=False,
                    focus_x=fx, focus_y=fy, fps=fps,
                )
                accumulated_scale = 1.0
            else:
                # Last zoom in a run – include hold + ease-out
                clip = _create_constant_velocity_zoom_clip(
                    img_path, seg_duration, target_size,
                    start_scale=accumulated_scale,
                    include_reset=True,
                    focus_x=fx, focus_y=fy, fps=fps,
                )
                accumulated_scale = 1.0
        elif effect == "zoom_out_slow":
            accumulated_scale = 1.0
            hold = seg_duration / 2.0
            clip = _create_zoom_clip_focused(img_path, seg_duration, target_size, "out",
                                             fps, hold_duration=hold,
                                             focus_x=fx, focus_y=fy)
        elif effect == "audio_reactive_shake":
            accumulated_scale = 1.0
            peak_offset = max(0.0, ev.get("peak_time", start) - start)
            clip = _create_shake_clip(img_path, seg_duration, target_size,
                                      peak_offset=peak_offset, fps=fps)
        elif effect == "pop_scale":
            accumulated_scale = 1.0
            clip = _create_pop_scale_clip_focused(img_path, seg_duration, target_size, fps,
                                                   focus_x=fx, focus_y=fy)
        else:
            # hard_cut or unrecognised → static
            accumulated_scale = 1.0
            clip = _create_static_clip(img_path, seg_duration, target_size, fps)

        if ev.get("color_grade"):
            grade = ev["color_grade"]
            clip = clip.image_transform(lambda frame, _g=grade: apply_color_shift(frame, _g))

        segments.append(clip)

    if not segments:
        segments.append(_create_static_clip(image_paths[0], total_duration, target_size, fps))

    final = concatenate_videoclips(segments, method="compose")
    final = final.with_audio(audio_clip)
    final.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac")
    return output_path
