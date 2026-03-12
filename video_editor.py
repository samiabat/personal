import random
import numpy as np
from PIL import Image
from moviepy import AudioFileClip, ImageClip, VideoClip, concatenate_videoclips


def _create_ken_burns_clip(image_path: str, duration: float, target_size: tuple,
                           enable_zoom: bool = True, enable_shake: bool = True, fps: int = 24):
    """Create a clip with optional gentle zoom and/or pan (shake) effect.

    Zoom and pan can be independently enabled.  When both are disabled the
    caller should use a plain ``ImageClip`` instead, but this function will
    still return a valid static clip for safety.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = target_size

    scale = max(w / img.width, h / img.height) * 1.08
    new_w, new_h = int(img.width * scale), int(img.height * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    img_array = np.array(img_resized)

    # Zoom parameters — only applied when enable_zoom is True
    if enable_zoom:
        zoom_in = random.choice([True, False])
        zoom_start = 1.0 if zoom_in else 1.04
        zoom_end = 1.04 if zoom_in else 1.0
    else:
        zoom_start = 1.0
        zoom_end = 1.0

    # Pan (shake) parameters — only applied when enable_shake is True
    if enable_shake:
        pan_x_range = (new_w - w) * 0.1
        pan_y_range = (new_h - h) * 0.05
        pan_x_start = random.uniform(-pan_x_range, pan_x_range) * 0.3
        pan_x_end = random.uniform(-pan_x_range, pan_x_range) * 0.3
        pan_y_start = random.uniform(-pan_y_range, pan_y_range) * 0.15
        pan_y_end = random.uniform(-pan_y_range, pan_y_range) * 0.15
    else:
        pan_x_start = pan_x_end = 0
        pan_y_start = pan_y_end = 0

    def make_frame(t):
        progress = t / max(duration, 0.001)
        progress = 0.5 - 0.5 * np.cos(progress * np.pi)

        current_zoom = zoom_start + (zoom_end - zoom_start) * progress
        crop_w = int(w / current_zoom)
        crop_h = int(h / current_zoom)

        cx = new_w // 2 + int(pan_x_start + (pan_x_end - pan_x_start) * progress)
        cy = new_h // 2 + int(pan_y_start + (pan_y_end - pan_y_start) * progress)

        x1 = max(0, min(cx - crop_w // 2, new_w - crop_w))
        y1 = max(0, min(cy - crop_h // 2, new_h - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = img_array[y1:y2, x1:x2]
        frame = Image.fromarray(cropped).resize((w, h), Image.LANCZOS)
        return np.array(frame)

    return VideoClip(make_frame, duration=duration).with_fps(fps)


def assemble_final_video(total_scenes: int, asset_folder: str, output_filename: str,
                         resolution: str = "1080p", enable_ken_burns: bool = False,
                         enable_zoom: bool = False, enable_shake: bool = False):
    """Takes all generated assets, stitches them, and renders the MP4.

    ``enable_ken_burns`` is kept for backward-compatibility and treated as
    enabling **both** zoom and shake when the individual flags are not set.
    """
    # Legacy compat: if old flag is True and new flags are both False, enable both
    if enable_ken_burns and not enable_zoom and not enable_shake:
        enable_zoom = True
        enable_shake = True

    res_map = {
        "480p": (854, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4K": (3840, 2160),
    }
    target_size = res_map.get(resolution, (1920, 1080))

    video_clips = []
    for index in range(total_scenes):
        audio_path = f"{asset_folder}/scene_{index}_audio.wav"
        image_path = f"{asset_folder}/scene_{index}_image.jpg"
        audio_clip = AudioFileClip(audio_path)

        if enable_zoom or enable_shake:
            video_clip = _create_ken_burns_clip(
                image_path, audio_clip.duration, target_size,
                enable_zoom=enable_zoom, enable_shake=enable_shake,
            )
            video_clip = video_clip.with_audio(audio_clip)
        else:
            image_clip = ImageClip(image_path).with_duration(audio_clip.duration).resized(target_size)
            video_clip = image_clip.with_audio(audio_clip)

        video_clips.append(video_clip)

    print("Concatenating all scenes...")
    final_video = concatenate_videoclips(video_clips, method="compose")
    output_path = f"{asset_folder}/{output_filename}"
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
    return output_path