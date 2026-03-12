import os
from script_content import scenes
from gemini_services import generate_audio
from image_services import generate_image
from video_editor import assemble_final_video

ASSET_DIR = "assets"

def setup_directories():
    if not os.path.exists(ASSET_DIR):
        os.makedirs(ASSET_DIR)

def cleanup_temp_files(total_scenes):
    print("Cleaning up temporary files...")
    for index in range(total_scenes):
        audio_path = f"{ASSET_DIR}/scene_{index}_audio.wav"
        image_path = f"{ASSET_DIR}/scene_{index}_image.jpg"
        if os.path.exists(audio_path): os.remove(audio_path)
        if os.path.exists(image_path): os.remove(image_path)

def main(
    image_provider: str = "gemini",
    image_model: str = None,
    together_size: str = "1024x576",
    enable_shake: bool = False,
    resume: bool = True,
):
    """Run the video generation pipeline.

    Parameters
    ----------
    image_provider : str
        One of "gemini", "openai", "together".
    image_model : str | None
        Model name for the chosen provider (None = provider default).
    together_size : str
        Resolution for Together AI images ("1024x576" or "1280x720").
    enable_shake : bool
        Apply a Ken Burns (slow pan/zoom) effect on images in the final video.
    resume : bool
        When True, skip scenes whose audio and image files already exist.
    """
    setup_directories()
    print("Starting Modular Video Pipeline...\n")
    
    for index, scene in enumerate(scenes):
        print(f"--- Processing Scene {index + 1}/{len(scenes)} ---")
        
        audio_path = f"{ASSET_DIR}/scene_{index}_audio.wav"
        image_path = f"{ASSET_DIR}/scene_{index}_image.jpg"
        
        # 1. Generate Audio (skip if resume is on and file exists)
        if resume and os.path.exists(audio_path):
            print(f"Audio already exists, skipping: {audio_path}")
        else:
            print("Generating voiceover...")
            generate_audio(scene["voiceover"], audio_path)

        # 2. Generate Image (skip if resume is on and file exists)
        if resume and os.path.exists(image_path):
            print(f"Image already exists, skipping: {image_path}")
        else:
            print("Generating image...")
            generate_image(
                scene["prompt"],
                image_path,
                provider=image_provider,
                model=image_model,
                together_size=together_size,
            )

        print(f"Assets for Scene {index + 1} generated successfully.\n")
        
    # 3. Assemble Video
    print("Building final video...")
    output_path = assemble_final_video(
        len(scenes), ASSET_DIR, "final_trading_short.mp4",
        enable_shake=enable_shake,
    )
    print(f"\nSUCCESS! Video saved as: {output_path}")
    
    # 4. Clean up (Optional, comment out if you want to keep the raw images/audio)
    cleanup_temp_files(len(scenes))

if __name__ == "__main__":
    main()