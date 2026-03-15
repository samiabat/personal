"""V2 Video Editing Engine.

Handles "Grouped Scene" assembly where one voiceover maps to multiple
images via visual beats with word-level timestamp synchronisation.
"""

import numpy as np
from PIL import Image
from moviepy import AudioFileClip, ImageClip, VideoClip, concatenate_videoclips


# ---------------------------------------------------------------------------
# Word-level timestamp helpers
# ---------------------------------------------------------------------------

def get_word_timestamps(audio_path: str, api_key: str = "") -> list[dict]:
    """Use OpenAI Whisper API to extract word-level timestamps from audio.

    Returns a list of dicts with ``word``, ``start`` and ``end`` keys.
    """
    from config import get_openai_client

    client = get_openai_client(api_key)
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
    # transcript.words is a list of objects with word/start/end attributes
    return [{"word": w.word, "start": w.start, "end": w.end} for w in transcript.words]


def find_trigger_timestamp(word_timestamps: list[dict], trigger_word: str) -> float:
    """Find the start-time of *trigger_word* in the word timestamp list.

    Supports multi-word triggers (e.g. ``"single choice"``).  Falls back to
    ``0.0`` when no match is found.
    """
    trigger_parts = trigger_word.lower().split()

    def _clean(w: str) -> str:
        return w.lower().strip(".,!?;:\"'")

    if len(trigger_parts) == 1:
        for w in word_timestamps:
            if trigger_parts[0] == _clean(w["word"]):
                return w["start"]
    else:
        for i, w in enumerate(word_timestamps):
            if _clean(w["word"]) == trigger_parts[0]:
                match = True
                for j, tp in enumerate(trigger_parts[1:], 1):
                    if i + j >= len(word_timestamps):
                        match = False
                        break
                    if _clean(word_timestamps[i + j]["word"]) != tp:
                        match = False
                        break
                if match:
                    return w["start"]
    return 0.0


# ---------------------------------------------------------------------------
# Audio analysis (librosa)
# ---------------------------------------------------------------------------

def find_nearest_audio_peak(audio_path: str, target_time: float,
                            search_range: float = 0.5) -> float:
    """Return the timestamp of the nearest volume peak to *target_time*.

    Uses librosa RMS energy to locate peaks within *search_range* seconds.
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=None)
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop)

    lo = max(0.0, target_time - search_range)
    hi = target_time + search_range
    mask = (times >= lo) & (times <= hi)

    if not mask.any():
        return target_time

    peak_idx = np.argmax(rms[mask])
    return float(times[mask][peak_idx])


# ---------------------------------------------------------------------------
# Cinematic zoom constants – consistent across all scenes
# ---------------------------------------------------------------------------

CINEMATIC_ZOOM_IN_DUR = 1.0    # seconds for smooth ease-in zoom
CINEMATIC_HOLD_DUR = 3.0       # seconds to hold peak scale
CINEMATIC_ZOOM_OUT_DUR = 1.0   # seconds for smooth ease-out back to 1.0
MIN_ZOOM_DURATION = 5.0        # minimum segment length to apply zoom effects
ZOOM_TARGET = 1.15             # peak zoom scale


# ---------------------------------------------------------------------------
# Easing helpers
# ---------------------------------------------------------------------------

def _smoothstep(t: float) -> float:
    """Hermite smooth-step interpolation: 3t² − 2t³.

    Maps *t* ∈ [0, 1] → [0, 1] with zero first-derivative at both ends,
    giving a perceptually "cinematic" ease-in / ease-out curve.
    """
    t = np.clip(t, 0.0, 1.0)
    return float(3.0 * t ** 2 - 2.0 * t ** 3)


# ---------------------------------------------------------------------------
# Effect helpers – each returns a MoviePy clip
# ---------------------------------------------------------------------------

def _prepare_image(image_path: str, target_size: tuple, extra_scale: float = 1.15):
    """Load & scale an image so it is larger than *target_size*."""
    img = Image.open(image_path).convert("RGB")
    w, h = target_size
    scale = max(w / img.width, h / img.height) * extra_scale
    new_w, new_h = int(img.width * scale), int(img.height * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(img_resized), new_w, new_h


def _create_static_clip(image_path: str, duration: float, target_size: tuple,
                         fps: int = 24) -> VideoClip:
    """Plain static image clip, centre-cropped to *target_size*."""
    img_arr, new_w, new_h = _prepare_image(image_path, target_size, 1.02)
    w, h = target_size

    cx, cy = new_w // 2, new_h // 2
    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    frame = img_arr[y1:y1 + h, x1:x1 + w]

    if frame.shape[0] != h or frame.shape[1] != w:
        frame = np.array(Image.fromarray(frame).resize((w, h), Image.LANCZOS))

    return ImageClip(frame).with_duration(duration).with_fps(fps)


def _create_zoom_clip(image_path: str, duration: float, target_size: tuple,
                       direction: str = "in", fps: int = 24,
                       hold_duration: float | None = None,
                       ease_duration: float = CINEMATIC_ZOOM_IN_DUR,
                       zoom_target: float = ZOOM_TARGET) -> VideoClip:
    """Cinematic state-based zoom with Ease-In → Hold → Ease-Out phases.

    Parameters
    ----------
    direction : str
        ``"in"``  → scale ramps *up* to *zoom_target* then back to 1.0.
        ``"out"`` → scale ramps *down* to 1.0 then back to *zoom_target*.
    hold_duration : float | None
        How long to hold the peak zoom.  When ``None`` the hold fills
        whatever time remains after the two ease phases.
    ease_duration : float
        Duration (seconds) of each ease phase (default 0.5 s).
    zoom_target : float
        Peak zoom scale (default 1.15).
    """
    img_arr, new_w, new_h = _prepare_image(image_path, target_size, 1.20)
    w, h = target_size

    # --- phase durations (clamp so they never exceed total duration) ---------
    max_phase_dur = duration / 3.0
    ease_in_dur = min(ease_duration, max_phase_dur)
    ease_out_dur = min(ease_duration, max_phase_dur)

    if hold_duration is not None:
        hold_dur = hold_duration
        # If explicit hold + eases exceed the segment, shrink hold first
        total = ease_in_dur + hold_dur + ease_out_dur
        if total > duration:
            hold_dur = max(0.0, duration - ease_in_dur - ease_out_dur)
    else:
        # Fill remaining time with the hold
        hold_dur = max(0.0, duration - ease_in_dur - ease_out_dur)

    # --- zoom range ----------------------------------------------------------
    if direction == "in":
        zoom_base = 1.0
        zoom_peak = zoom_target
    else:
        zoom_base = zoom_target
        zoom_peak = 1.0

    def make_frame(t):
        if t < ease_in_dur:
            # Phase 1 – Ease-In
            progress = _smoothstep(t / max(ease_in_dur, 1e-6))
            zoom = zoom_base + (zoom_peak - zoom_base) * progress
        elif t < ease_in_dur + hold_dur:
            # Phase 2 – Hold
            zoom = zoom_peak
        else:
            # Phase 3 – Ease-Out
            ease_out_t = t - ease_in_dur - hold_dur
            progress = _smoothstep(ease_out_t / max(ease_out_dur, 1e-6))
            zoom = zoom_peak + (zoom_base - zoom_peak) * progress

        crop_w = int(w / zoom)
        crop_h = int(h / zoom)

        cx, cy = new_w // 2, new_h // 2
        x1 = max(0, min(cx - crop_w // 2, new_w - crop_w))
        y1 = max(0, min(cy - crop_h // 2, new_h - crop_h))

        cropped = img_arr[y1:y1 + crop_h, x1:x1 + crop_w]
        return np.array(Image.fromarray(cropped).resize((w, h), Image.LANCZOS))

    return VideoClip(make_frame, duration=duration).with_fps(fps)


def _create_shake_clip(image_path: str, duration: float, target_size: tuple,
                        peak_offset: float = 0.0, shake_duration: float = 0.2,
                        intensity: int = 5, fps: int = 24) -> VideoClip:
    """Audio-reactive positional jitter at *peak_offset* within the clip."""
    img_arr, new_w, new_h = _prepare_image(image_path, target_size, 1.08)
    w, h = target_size

    def make_frame(t):
        cx, cy = new_w // 2, new_h // 2

        if peak_offset <= t <= peak_offset + shake_duration:
            progress = (t - peak_offset) / shake_duration
            dampen = 1.0 - progress
            dx = int(np.sin(t * 50) * intensity * dampen)
            dy = int(np.cos(t * 37) * intensity * dampen)
            cx += dx
            cy += dy

        x1 = max(0, min(cx - w // 2, new_w - w))
        y1 = max(0, min(cy - h // 2, new_h - h))
        cropped = img_arr[y1:y1 + h, x1:x1 + w]

        if cropped.shape[0] != h or cropped.shape[1] != w:
            cropped = np.array(Image.fromarray(cropped).resize((w, h), Image.LANCZOS))
        return cropped

    return VideoClip(make_frame, duration=duration).with_fps(fps)


def _create_pop_scale_clip(image_path: str, duration: float, target_size: tuple,
                            fps: int = 24) -> VideoClip:
    """Quick zoom-pop then settle back to 1×."""
    img_arr, new_w, new_h = _prepare_image(image_path, target_size, 1.20)
    w, h = target_size
    pop_dur = min(0.3, duration * 0.5)

    def make_frame(t):
        if t < pop_dur:
            progress = t / pop_dur
            zoom = 1.0 + 0.08 * np.sin(progress * np.pi)
        else:
            zoom = 1.0

        crop_w = int(w / zoom)
        crop_h = int(h / zoom)
        cx, cy = new_w // 2, new_h // 2
        x1 = max(0, min(cx - crop_w // 2, new_w - crop_w))
        y1 = max(0, min(cy - crop_h // 2, new_h - crop_h))

        cropped = img_arr[y1:y1 + crop_h, x1:x1 + crop_w]
        return np.array(Image.fromarray(cropped).resize((w, h), Image.LANCZOS))

    return VideoClip(make_frame, duration=duration).with_fps(fps)


# ---------------------------------------------------------------------------
# Colour grading
# ---------------------------------------------------------------------------

def apply_color_shift(frame: np.ndarray, color_grade: str) -> np.ndarray:
    """Apply a colour-grade transformation to a single frame array."""
    if color_grade == "dark":
        return np.clip(frame.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    if color_grade == "warm":
        f = frame.copy().astype(np.float32)
        f[:, :, 0] = np.clip(f[:, :, 0] * 1.1, 0, 255)
        f[:, :, 2] = np.clip(f[:, :, 2] * 0.9, 0, 255)
        return f.astype(np.uint8)
    if color_grade == "cool":
        f = frame.copy().astype(np.float32)
        f[:, :, 0] = np.clip(f[:, :, 0] * 0.9, 0, 255)
        f[:, :, 2] = np.clip(f[:, :, 2] * 1.1, 0, 255)
        return f.astype(np.uint8)
    if color_grade == "high_contrast":
        mean = frame.mean()
        return np.clip((frame.astype(np.float32) - mean) * 1.3 + mean, 0, 255).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Beat validation helpers
# ---------------------------------------------------------------------------

ZOOM_EFFECTS = {"zoom_in_slow", "zoom_out_slow"}
SCALE_EFFECTS = {"zoom_in_slow", "zoom_out_slow", "pop_scale"}
MIN_EFFECT_GAP = 2.0  # minimum seconds between effects on the same image


def _sanitize_beats(beat_events: list[dict]) -> list[dict]:
    """Enforce effect constraints on resolved beat events.

    Rules
    -----
    1. At most **one** zoom effect (``zoom_in_slow`` / ``zoom_out_slow``)
       per ``image_index``.
    2. Scale-based effects (``zoom_*`` / ``pop_scale``) on the **same** image
       must be separated by at least ``MIN_EFFECT_GAP`` seconds; the later
       duplicate is downgraded to ``hard_cut``.
    """
    # Track zoom usage per image index
    image_zoom_used: dict[int, bool] = {}
    # Track last scale-effect time per image index
    image_last_scale_time: dict[int, float] = {}

    sanitized = []
    for ev in beat_events:
        img_idx = ev["image_index"]
        effect = ev["effect"]
        t = ev["time"]

        if effect in ZOOM_EFFECTS:
            if image_zoom_used.get(img_idx, False):
                # Already have a zoom on this image → downgrade to static cut
                ev = {**ev, "effect": "hard_cut"}
            else:
                image_zoom_used[img_idx] = True

        # Check minimum gap between scale effects on same image
        if ev["effect"] in SCALE_EFFECTS:
            last_t = image_last_scale_time.get(img_idx)
            if last_t is not None and (t - last_t) < MIN_EFFECT_GAP:
                ev = {**ev, "effect": "hard_cut"}
            else:
                image_last_scale_time[img_idx] = t

        sanitized.append(ev)

    return sanitized


# ---------------------------------------------------------------------------
# V2 assembly
# ---------------------------------------------------------------------------

def assemble_v2_video(image_paths: list[str], audio_path: str,
                      visual_beats: list[dict], word_timestamps: list[dict],
                      output_path: str, target_size: tuple = (1920, 1080),
                      fps: int = 24,
                      enable_subtitles: bool = False,
                      subtitle_style: str = "cinematic") -> str:
    """Build the final V2 video from images, one audio track & visual beats.

    Parameters
    ----------
    image_paths : list[str]
        Ordered list of image file paths (one per prompt).
    audio_path : str
        Path to the full voiceover WAV.
    visual_beats : list[dict]
        Each dict has ``trigger_word``, ``effect``, ``image_index`` and
        optional ``color_grade`` and ``hold_duration`` (seconds).  When
        ``hold_duration`` is omitted it defaults to
        ``(next_beat_time − current_beat_time) / 2``.
    word_timestamps : list[dict]
        Word-level timestamps from Whisper (``word``, ``start``, ``end``).
    output_path : str
        Where to write the final MP4.
    target_size : tuple
        ``(width, height)`` of the output.
    fps : int
        Frames per second.
    """
    audio_clip = AudioFileClip(audio_path)
    total_duration = audio_clip.duration

    # Resolve beat timestamps ------------------------------------------------
    beat_events = []
    for beat in visual_beats:
        t = find_trigger_timestamp(word_timestamps, beat["trigger_word"])
        beat_events.append({
            "time": t,
            "effect": beat["effect"],
            "image_index": beat["image_index"],
            "color_grade": beat.get("color_grade"),
            "hold_duration": beat.get("hold_duration"),
        })

    beat_events.sort(key=lambda x: x["time"])

    # Enforce one-zoom-per-image and minimum spacing between effects -------
    beat_events = _sanitize_beats(beat_events)

    # For shake effects, snap to nearest audio peak --------------------------
    for ev in beat_events:
        if ev["effect"] == "audio_reactive_shake":
            ev["peak_time"] = find_nearest_audio_peak(audio_path, ev["time"])

    # Build clip segments ----------------------------------------------------
    segments: list[VideoClip] = []

    # Optional pre-first-beat static segment
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

        if effect == "zoom_in_slow":
            if seg_duration < MIN_ZOOM_DURATION:
                clip = _create_static_clip(img_path, seg_duration, target_size, fps)
            else:
                hold = ev["hold_duration"]
                if hold is None:
                    hold = CINEMATIC_HOLD_DUR
                clip = _create_zoom_clip(img_path, seg_duration, target_size, "in",
                                          fps, hold_duration=hold)
        elif effect == "zoom_out_slow":
            if seg_duration < MIN_ZOOM_DURATION:
                clip = _create_static_clip(img_path, seg_duration, target_size, fps)
            else:
                hold = ev["hold_duration"]
                if hold is None:
                    hold = CINEMATIC_HOLD_DUR
                clip = _create_zoom_clip(img_path, seg_duration, target_size, "out",
                                          fps, hold_duration=hold)
        elif effect == "audio_reactive_shake":
            peak_offset = max(0.0, ev.get("peak_time", start) - start)
            clip = _create_shake_clip(img_path, seg_duration, target_size,
                                      peak_offset=peak_offset, fps=fps)
        elif effect == "pop_scale":
            clip = _create_pop_scale_clip(img_path, seg_duration, target_size, fps)
        else:
            # hard_cut or any unrecognised effect → static
            clip = _create_static_clip(img_path, seg_duration, target_size, fps)

        # Colour grading
        if ev.get("color_grade"):
            grade = ev["color_grade"]
            clip = clip.image_transform(lambda frame, _g=grade: apply_color_shift(frame, _g))

        segments.append(clip)

    # Concatenate & render ---------------------------------------------------
    if not segments:
        # Fallback: single static clip from first image
        segments.append(_create_static_clip(image_paths[0], total_duration, target_size, fps))

    final = concatenate_videoclips(segments, method="compose")
    final = final.with_audio(audio_clip)

    if enable_subtitles:
        from subtitle_renderer import render_subtitles
        final = render_subtitles(final, word_timestamps, target_size, subtitle_style)

    final.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac")
    return output_path
