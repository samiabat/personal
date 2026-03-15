"""Fancy subtitle / text overlay renderer.

Renders word-level subtitles on top of a MoviePy video clip using PIL for
high-quality text with shadows, outlines, and styling.

Supported styles
----------------
* ``"cinematic"`` – Large centred white text with a dark shadow, displayed
  as phrase groups (3–5 words at a time).
* ``"minimal"`` – Smaller bottom-aligned text with a semi-transparent
  background bar.
* ``"typewriter"`` – Words appear one-by-one with a highlight on the
  current word.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoClip, CompositeVideoClip


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

def _get_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Load the best available font at the requested size."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    if not bold:
        font_paths = [p.replace("-Bold", "") for p in font_paths] + font_paths

    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Phrase grouping
# ---------------------------------------------------------------------------

def _group_words_into_phrases(word_timestamps: list[dict],
                              words_per_phrase: int = 4) -> list[dict]:
    """Group word timestamps into phrases of *words_per_phrase* words.

    Returns a list of dicts with ``text``, ``start``, and ``end`` keys.
    """
    phrases = []
    for i in range(0, len(word_timestamps), words_per_phrase):
        chunk = word_timestamps[i:i + words_per_phrase]
        text = " ".join(w["word"].strip() for w in chunk)
        phrases.append({
            "text": text,
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
        })
    return phrases


# ---------------------------------------------------------------------------
# Style renderers
# ---------------------------------------------------------------------------

def _render_cinematic_frame(text: str, size: tuple) -> np.ndarray:
    """Render a centred cinematic subtitle with shadow."""
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_size = max(28, int(h * 0.045))
    font = _get_font(font_size, bold=True)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - tw) // 2
    y = int(h * 0.82)

    # Shadow
    shadow_offset = max(2, font_size // 16)
    draw.text((x + shadow_offset, y + shadow_offset), text,
              font=font, fill=(0, 0, 0, 200))
    # Main text
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    return np.array(img)


def _render_minimal_frame(text: str, size: tuple) -> np.ndarray:
    """Render a bottom-bar subtitle with semi-transparent background."""
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_size = max(22, int(h * 0.035))
    font = _get_font(font_size, bold=False)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - tw) // 2
    y = int(h * 0.90)

    pad = 10
    draw.rounded_rectangle(
        [x - pad, y - pad, x + tw + pad, y + th + pad],
        radius=8,
        fill=(0, 0, 0, 160),
    )
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 240))

    return np.array(img)


def _render_typewriter_frame(words: list[dict], current_time: float,
                             size: tuple) -> np.ndarray:
    """Render words with the current word highlighted."""
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_size = max(28, int(h * 0.045))
    font = _get_font(font_size, bold=True)
    font_highlight = _get_font(font_size, bold=True)

    # Find current visible words (show surrounding context)
    current_idx = 0
    for i, wd in enumerate(words):
        if wd["start"] <= current_time <= wd["end"]:
            current_idx = i
            break
        if wd["start"] > current_time:
            current_idx = max(0, i - 1)
            break
    else:
        current_idx = len(words) - 1

    # Show a window of words around the current word
    window = 5
    start_i = max(0, current_idx - window // 2)
    end_i = min(len(words), start_i + window)
    visible_words = words[start_i:end_i]

    full_text = " ".join(wd["word"].strip() for wd in visible_words)
    bbox = draw.textbbox((0, 0), full_text, font=font)
    total_w = bbox[2] - bbox[0]
    x_start = (w - total_w) // 2
    y = int(h * 0.82)

    # Shadow backdrop
    shadow_offset = max(2, font_size // 16)
    draw.text((x_start + shadow_offset, y + shadow_offset), full_text,
              font=font, fill=(0, 0, 0, 180))

    # Render each word, highlighting the current one
    x_cursor = x_start
    for wd in visible_words:
        word_text = wd["word"].strip()
        is_current = wd["start"] <= current_time <= wd["end"]

        if is_current:
            color = (255, 220, 80, 255)  # Gold highlight
        else:
            color = (255, 255, 255, 220)

        draw.text((x_cursor, y), word_text, font=font_highlight if is_current else font,
                  fill=color)
        word_bbox = draw.textbbox((0, 0), word_text + " ", font=font)
        x_cursor += word_bbox[2] - word_bbox[0]

    return np.array(img)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def render_subtitles(video_clip: VideoClip, word_timestamps: list[dict],
                     target_size: tuple,
                     style: str = "cinematic") -> CompositeVideoClip:
    """Overlay subtitles on *video_clip* using word-level timestamps.

    Parameters
    ----------
    video_clip : VideoClip
        The base video to overlay subtitles on.
    word_timestamps : list[dict]
        Word-level timestamps (``word``, ``start``, ``end``).
    target_size : tuple
        ``(width, height)`` of the video.
    style : str
        One of ``"cinematic"``, ``"minimal"``, or ``"typewriter"``.

    Returns
    -------
    CompositeVideoClip
        The video with subtitle overlay composited on top.
    """
    if not word_timestamps:
        return video_clip

    duration = video_clip.duration

    if style == "typewriter":
        # Typewriter needs per-frame word lookup
        def make_subtitle_frame(t):
            return _render_typewriter_frame(word_timestamps, t, target_size)

        subtitle_clip = VideoClip(make_subtitle_frame, duration=duration)
        subtitle_clip = subtitle_clip.with_fps(video_clip.fps or 24)
        return CompositeVideoClip([video_clip, subtitle_clip])

    # Phrase-based styles (cinematic, minimal)
    phrases = _group_words_into_phrases(word_timestamps,
                                        words_per_phrase=4 if style == "cinematic" else 5)

    renderer = _render_cinematic_frame if style == "cinematic" else _render_minimal_frame

    def make_subtitle_frame(t):
        for phrase in phrases:
            if phrase["start"] <= t <= phrase["end"]:
                return renderer(phrase["text"], target_size)
        # No active phrase → transparent
        return np.zeros((*target_size[::-1], 4), dtype=np.uint8)

    subtitle_clip = VideoClip(make_subtitle_frame, duration=duration)
    subtitle_clip = subtitle_clip.with_fps(video_clip.fps or 24)
    return CompositeVideoClip([video_clip, subtitle_clip])
