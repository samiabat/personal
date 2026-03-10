import os
from script_content import scenes
from gemini_services import generate_audio, generate_image
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

def main():
    setup_directories()
    print("Starting Modular Video Pipeline...\n")
    
    for index, scene in enumerate(scenes):
        print(f"--- Processing Scene {index + 1}/{len(scenes)} ---")
        
        
        audio_path = f"{ASSET_DIR}/scene_{index}_audio.wav"
        image_path = f"{ASSET_DIR}/scene_{index}_image.jpg"
        
        # 1. Generate Audio
        # if index > 21:
        print("Generating voiceover...")
        generate_audio(scene["voiceover"], audio_path)
        
        # 2. Generate Image
        print("Generating image...")
        generate_image(scene["prompt"], image_path)
        print(f"Assets for Scene {index + 1} generated successfully.\n")
        
    # 3. Assemble Video
    print("Building final video...")
    output_path = assemble_final_video(len(scenes), ASSET_DIR, "final_trading_short.mp4")
    print(f"\nSUCCESS! Video saved as: {output_path}")
    
    # 4. Clean up (Optional, comment out if you want to keep the raw images/audio)
    cleanup_temp_files(len(scenes))

if __name__ == "__main__":
    main()