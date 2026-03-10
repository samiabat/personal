from moviepy import AudioFileClip, ImageClip, concatenate_videoclips

def assemble_final_video(total_scenes: int, asset_folder: str, output_filename: str):
    """Takes all generated assets, stitches them, and renders the MP4."""
    video_clips = []
    
    for index in range(total_scenes):
        audio_path = f"{asset_folder}/scene_{index}_audio.wav"
        image_path = f"{asset_folder}/scene_{index}_image.jpg"
        
        audio_clip = AudioFileClip(audio_path)
        image_clip = ImageClip(image_path).with_duration(audio_clip.duration)
        video_clip = image_clip.with_audio(audio_clip)
        
        video_clips.append(video_clip)
        
    print("Concatenating all scenes...")
    final_video = concatenate_videoclips(video_clips, method="compose")
    
    output_path = f"{asset_folder}/{output_filename}"
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
    return output_path