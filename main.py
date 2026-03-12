import os
import argparse
from script_content import scenes
from gemini_services import generate_audio, generate_image_with_provider
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

def _asset_exists(path: str) -> bool:
    """Check if an asset file exists and is non-empty."""
    return os.path.exists(path) and os.path.getsize(path) > 0

def parse_image_size(size_str: str):
    """Parse image size string like '1024x576' into (width, height)."""
    parts = size_str.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid size format '{size_str}'. Use WIDTHxHEIGHT (e.g., 1024x576)")
    return int(parts[0]), int(parts[1])

def main():
    parser = argparse.ArgumentParser(description="Modular Video Pipeline")
    parser.add_argument("--image-provider", choices=["gemini", "openai", "togetherai"],
                        default="gemini",
                        help="Image generation provider (default: gemini)")
    parser.add_argument("--image-model", type=str, default=None,
                        help="Image model name. Defaults: gemini='gemini-3.1-flash-image-preview', "
                             "openai='gpt-image-1', togetherai='black-forest-labs/FLUX.1-schnell'. "
                             "OpenAI options: gpt-image-1-mini, gpt-image-1, gpt-image-1.5. "
                             "Together AI options: black-forest-labs/FLUX.1-schnell, "
                             "Lykon/dreamshaper-xl-v2-turbo, etc.")
    parser.add_argument("--image-size", type=str, default="1024x576",
                        help="Image size for Together AI as WIDTHxHEIGHT (default: 1024x576). "
                             "Common options: 1024x576, 1280x720")
    parser.add_argument("--ken-burns", action="store_true",
                        help="Enable Ken Burns effect (slow zoom/pan) on images in the video")
    parser.add_argument("--no-resume", action="store_true",
                        help="Regenerate all assets even if they already exist")
    args = parser.parse_args()

    width, height = parse_image_size(args.image_size)
    resume = not args.no_resume

    setup_directories()
    print("Starting Modular Video Pipeline...\n")
    print(f"Image Provider: {args.image_provider}")
    if args.image_model:
        print(f"Image Model: {args.image_model}")
    if args.image_provider == "togetherai":
        print(f"Image Size: {width}x{height}")
    if args.ken_burns:
        print("Ken Burns Effect: ENABLED")
    if resume:
        print("Resume Mode: ON (skipping existing assets)")
    print()
    
    for index, scene in enumerate(scenes):
        print(f"--- Processing Scene {index + 1}/{len(scenes)} ---")
        
        audio_path = f"{ASSET_DIR}/scene_{index}_audio.wav"
        image_path = f"{ASSET_DIR}/scene_{index}_image.jpg"
        
        # 1. Generate Audio (skip if already exists and resume is on)
        if resume and _asset_exists(audio_path):
            print(f"Audio already exists, skipping: {audio_path}")
        else:
            print("Generating voiceover...")
            generate_audio(scene["voiceover"], audio_path)
        
        # 2. Generate Image (skip if already exists and resume is on)
        if resume and _asset_exists(image_path):
            print(f"Image already exists, skipping: {image_path}")
        else:
            print("Generating image...")
            generate_image_with_provider(
                scene["prompt"], image_path,
                provider=args.image_provider,
                model=args.image_model,
                width=width, height=height,
            )

        print(f"Assets for Scene {index + 1} ready.\n")
        
    # 3. Assemble Video
    print("Building final video...")
    output_path = assemble_final_video(len(scenes), ASSET_DIR, "final_trading_short.mp4",
                                        ken_burns=args.ken_burns)
    print(f"\nSUCCESS! Video saved as: {output_path}")
    
    # 4. Clean up (Optional, comment out if you want to keep the raw images/audio)
    cleanup_temp_files(len(scenes))

if __name__ == "__main__":
    main()