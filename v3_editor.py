"""V3 Video Editing Engine.

Extends the V2 pipeline with *focus-point-aware* zoom and pop-scale effects.
When ``focus_x`` / ``focus_y`` are present on a visual beat the camera
centres on that point instead of the default image centre.
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
# Focus-aware effect helpers
# ---------------------------------------------------------------------------

def _create_zoom_clip_focused(image_path: str, duration: float,
                               target_size: tuple, direction: str = "in",
                               fps: int = 24, hold_duration: float | None = None,
                               ease_duration: float = 0.5,
                               zoom_target: float = 1.15,
                               focus_x: float = 0.5,
                               focus_y: float = 0.5) -> VideoClip:
    """Cinematic zoom with optional focus point.

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
    fcx = int(np.clip(focus_x, 0.0, 1.0) * new_w)
    fcy = int(np.clip(focus_y, 0.0, 1.0) * new_h)

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

    fcx = int(np.clip(focus_x, 0.0, 1.0) * new_w)
    fcy = int(np.clip(focus_y, 0.0, 1.0) * new_h)

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
    """Build the final V3 video – identical to V2 but uses focus-point-aware
    zoom / pop-scale clips when ``focus_x`` / ``focus_y`` are present.

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
            "hold_duration": beat.get("hold_duration"),
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
            hold = ev["hold_duration"]
            if hold is None:
                hold = seg_duration / 2.0
            clip = _create_zoom_clip_focused(img_path, seg_duration, target_size, "in",
                                             fps, hold_duration=hold,
                                             focus_x=fx, focus_y=fy)
        elif effect == "zoom_out_slow":
            hold = ev["hold_duration"]
            if hold is None:
                hold = seg_duration / 2.0
            clip = _create_zoom_clip_focused(img_path, seg_duration, target_size, "out",
                                             fps, hold_duration=hold,
                                             focus_x=fx, focus_y=fy)
        elif effect == "audio_reactive_shake":
            peak_offset = max(0.0, ev.get("peak_time", start) - start)
            clip = _create_shake_clip(img_path, seg_duration, target_size,
                                      peak_offset=peak_offset, fps=fps)
        elif effect == "pop_scale":
            clip = _create_pop_scale_clip_focused(img_path, seg_duration, target_size, fps,
                                                   focus_x=fx, focus_y=fy)
        else:
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
