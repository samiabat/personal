import numpy as np
from moviepy import AudioFileClip, ImageClip, VideoClip, concatenate_videoclips


def _apply_ken_burns(image_clip, zoom_ratio=0.04):
    """Apply a slow Ken Burns (zoom + pan) effect to an image clip.

    The image is first scaled up slightly so that panning never reveals
    black edges.  A gentle pan direction is randomly chosen per clip so
    consecutive scenes feel varied.
    """
    w, h = image_clip.size
    duration = image_clip.duration

    # Base zoom that provides enough margin for movement
    base_zoom = 1.0 + zoom_ratio
    # End zoom – slightly more so there is a continuous slow zoom-in
    end_zoom = base_zoom + zoom_ratio

    # Random pan offsets (dx, dy) in pixels – will linearly interpolate
    np.random.seed(hash(str(duration)) % (2**31))
    max_pan_x = int(w * zoom_ratio * 0.5)
    max_pan_y = int(h * zoom_ratio * 0.5)
    start_dx = np.random.randint(-max_pan_x, max_pan_x + 1)
    start_dy = np.random.randint(-max_pan_y, max_pan_y + 1)
    end_dx = np.random.randint(-max_pan_x, max_pan_x + 1)
    end_dy = np.random.randint(-max_pan_y, max_pan_y + 1)

    src_img = image_clip.get_frame(0)
    src_h, src_w = src_img.shape[:2]

    def make_frame(t):
        progress = t / duration if duration > 0 else 0
        current_zoom = base_zoom + (end_zoom - base_zoom) * progress
        dx = start_dx + (end_dx - start_dx) * progress
        dy = start_dy + (end_dy - start_dy) * progress

        new_w = int(src_w / current_zoom)
        new_h = int(src_h / current_zoom)

        cx = src_w / 2 + dx
        cy = src_h / 2 + dy

        x1 = int(max(0, min(cx - new_w / 2, src_w - new_w)))
        y1 = int(max(0, min(cy - new_h / 2, src_h - new_h)))
        x2 = x1 + new_w
        y2 = y1 + new_h

        cropped = src_img[y1:y2, x1:x2]

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(cropped)
        pil_img = pil_img.resize((src_w, src_h), PILImage.LANCZOS)
        return np.array(pil_img)

    return VideoClip(make_frame, duration=duration).with_fps(24)


def assemble_final_video(
    total_scenes: int,
    asset_folder: str,
    output_filename: str,
    enable_shake: bool = False,
):
    """Takes all generated assets, stitches them, and renders the MP4.

    Parameters
    ----------
    enable_shake : bool
        When *True*, applies a subtle Ken Burns (zoom + pan) effect on each
        image so the final video has gentle motion instead of static stills.
    """
    video_clips = []

    for index in range(total_scenes):
        audio_path = f"{asset_folder}/scene_{index}_audio.wav"
        image_path = f"{asset_folder}/scene_{index}_image.jpg"

        audio_clip = AudioFileClip(audio_path)
        image_clip = ImageClip(image_path).with_duration(audio_clip.duration)

        if enable_shake:
            image_clip = _apply_ken_burns(image_clip)

        video_clip = image_clip.with_audio(audio_clip)
        video_clips.append(video_clip)

    print("Concatenating all scenes...")
    final_video = concatenate_videoclips(video_clips, method="compose")

    output_path = f"{asset_folder}/{output_filename}"
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
    return output_path