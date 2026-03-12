# Video Pipeline Dashboard

AI-powered video generation pipeline that creates narrated videos from scene scripts. Uses AI to generate voiceover audio and illustrations for each scene, then stitches everything into a final MP4.

## Features

- **Multi-provider image generation** – switch between Gemini, OpenAI, and Together AI
- **Resume on failure** – if the pipeline crashes mid-run, re-running picks up where it left off instead of regenerating completed scenes
- **Ken Burns effect** – optional slow pan/zoom motion on images so the video feels dynamic instead of static
- **Web dashboard** – configure everything from the browser: choose providers, models, resolutions, and toggle features on/off

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/samiabat/personal.git
cd personal
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create a `.env` file

Create a `.env` file in the project root with the API keys you have. You need **at least one** image provider key and the Gemini key (used for audio):

```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key          # optional
TOGETHER_API_KEY=your_together_api_key      # optional
```

> Audio generation always uses Gemini, so `GEMINI_API_KEY` is required. Image generation can use any of the three providers.

### 3. Run the web dashboard

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## Using the Dashboard

### Image Generation

Pick your provider and model from the **Image Generation** card:

| Provider | Models | Notes |
|----------|--------|-------|
| **Gemini** | `gemini-3.1-flash-image-preview` | Default provider |
| **OpenAI** | `gpt-image-1-mini`, `gpt-image-1`, `gpt-image-1.5` | Requires `OPENAI_API_KEY` |
| **Together AI** | `flux.1-schnell`, `dreamshaper`, `flux.1-dev`, `stable-diffusion-xl` | Requires `TOGETHER_API_KEY` |

For Together AI you can also choose the output resolution: **1024×576** or **1280×720**.

### Pipeline Options

| Toggle | What it does |
|--------|-------------|
| **Resume from failure** | Skips scenes whose audio and image files already exist in `assets/`. Turn this on so a failed run continues from where it stopped. Enabled by default. |
| **Ken Burns effect** | Adds a subtle slow zoom and pan on each image so the final video has gentle motion instead of static stills. Off by default. |

### Generating

1. Choose your provider, model, and toggles.
2. Click **Start Pipeline**.
3. The progress bar and scene list update automatically while the pipeline runs.
4. When finished, the video is saved to `assets/final_trading_short.mp4`.

## Running from the Command Line (optional)

You can also run the pipeline directly without the web UI:

```bash
python main.py
```

To customise options from code, call `main()` with arguments:

```python
from main import main

main(
    image_provider="together",       # "gemini", "openai", or "together"
    image_model="flux.1-schnell",    # model name for chosen provider
    together_size="1280x720",        # Together AI resolution
    enable_shake=True,               # Ken Burns effect
    resume=True,                     # skip already-generated assets
)
```

## Project Structure

```
├── app.py               # Flask web frontend (dashboard)
├── templates/
│   └── index.html       # Dashboard HTML/CSS/JS
├── main.py              # Pipeline orchestrator
├── config.py            # API key loading and client setup
├── gemini_services.py   # Audio (TTS) generation via Gemini
├── image_services.py    # Image generation (Gemini / OpenAI / Together AI)
├── video_editor.py      # MoviePy video assembly + Ken Burns effect
├── script_content.py    # Scene data (voiceover text + image prompts)
├── requirements.txt     # Python dependencies
└── assets/              # Generated audio, images, and final video (gitignored)
```

## Requirements

- Python 3.10+
- ffmpeg (required by MoviePy for video encoding)

Install ffmpeg if you don't have it:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Windows (via choco)
choco install ffmpeg
```
