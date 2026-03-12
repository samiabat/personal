import numpy as np
from PIL import Image
from moviepy import AudioFileClip, ImageClip, VideoClip, concatenate_videoclips


def _create_ken_burns_clip(image_path, duration, scene_index):
    """Creates a clip with a slow zoom + pan (Ken Burns) effect.
    
    Alternates between zoom-in/pan-right and zoom-out/pan-left
    for visual variety. The image is always slightly zoomed so 
    that panning never reveals black borders.
    """
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    aspect = img_w / img_h

    # Alternate direction per scene for variety
    if scene_index % 2 == 0:
        # Slow zoom in + pan left to right
        start_zoom, end_zoom = 1.12, 1.04
        start_x_ratio, end_x_ratio = 0.2, 0.8
    else:
        # Slow zoom out + pan right to left
        start_zoom, end_zoom = 1.04, 1.12
        start_x_ratio, end_x_ratio = 0.8, 0.2

    def make_frame(t):
        progress = t / max(duration, 0.001)

        # Interpolate zoom level
        zoom = start_zoom + (end_zoom - start_zoom) * progress

        # Visible crop dimensions (smaller than full image due to zoom)
        crop_w = img_w / zoom
        crop_h = crop_w / aspect
        if crop_h > img_h / zoom:
            crop_h = img_h / zoom
            crop_w = crop_h * aspect

        # Pan position
        max_pan_x = img_w - crop_w
        max_pan_y = img_h - crop_h
        x_ratio = start_x_ratio + (end_x_ratio - start_x_ratio) * progress
        pan_x = max_pan_x * x_ratio
        pan_y = max_pan_y * 0.5  # Center vertically

        # Clamp crop bounds within image
        left = max(0, min(int(pan_x), int(img_w - crop_w)))
        top = max(0, min(int(pan_y), int(img_h - crop_h)))
        right = left + int(crop_w)
        bottom = top + int(crop_h)

        cropped = img.crop((left, top, right, bottom))
        resized = cropped.resize((img_w, img_h), Image.LANCZOS)
        return np.array(resized)

    return VideoClip(make_frame, duration=duration)


def assemble_final_video(total_scenes: int, asset_folder: str, output_filename: str,
                         ken_burns: bool = False):
    """Takes all generated assets, stitches them, and renders the MP4.
    
    Args:
        total_scenes: Number of scenes to assemble.
        asset_folder: Folder containing scene audio/image files.
        output_filename: Name of the output MP4 file.
        ken_burns: If True, applies a slow zoom/pan effect on images.
    """
    video_clips = []
    
    for index in range(total_scenes):
        audio_path = f"{asset_folder}/scene_{index}_audio.wav"
        image_path = f"{asset_folder}/scene_{index}_image.jpg"
        
        audio_clip = AudioFileClip(audio_path)

        if ken_burns:
            image_clip = _create_ken_burns_clip(image_path, audio_clip.duration, index)
        else:
            image_clip = ImageClip(image_path).with_duration(audio_clip.duration)

        video_clip = image_clip.with_audio(audio_clip)
        video_clips.append(video_clip)
        
    print("Concatenating all scenes...")
    final_video = concatenate_videoclips(video_clips, method="compose")
    
    output_path = f"{asset_folder}/{output_filename}"
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
    return output_path