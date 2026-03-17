# 🎬 AI Video Generator

A full-stack web application that generates AI-powered videos from text scenes. Uses Google Gemini, OpenAI, ElevenLabs, and Together AI to create voiceovers and images, then assembles them into a downloadable MP4 video.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

## ✨ Features

- **🎥 Video Generation** — Define scenes with voiceover text and image prompts, generate a complete MP4 video
- **🔊 Multiple Speech Providers** — Choose between Google Gemini TTS, OpenAI TTS, or **ElevenLabs** ultra-realistic TTS (default)
- **🖼️ Flexible Image Models** — Select from preset image models or enter a custom model name; **Together AI** FLUX models set as default
- **🎙️ Voice Selection** — Pick from available voices or type a custom voice name
- **📐 Resolution Control** — Choose output resolution (480p, 720p, 1080p, 1440p, 4K)
- **📊 Real-time Progress** — Track generation progress with live updates via Server-Sent Events
- **⬇️ Download** — Download completed videos directly from the browser
- **🧪 Test Mode** — Test audio and image generation individually before committing to a full video
- **📝 Scene Editor** — Paste scenes as JSON or load built-in defaults
- **📼 My Videos** — Browse and re-download all previously generated videos; re-render with updated settings without regenerating assets
- **🔁 Asset Reuse** — Generated voiceovers and images are preserved after assembly so you can adjust settings and re-render at any time
- **⏱️ Time-Fit Editor** — For V5/V6 video clip scenes, change the time-fit strategy (auto / trim / slow-mo / loop) and re-render without regenerating audio

## 📁 Project Structure

```
├── app.py                 # FastAPI backend application
├── config.py              # API key configuration
├── gemini_services.py     # Google Gemini audio & image generation
├── openai_services.py     # OpenAI TTS generation
├── elevenlabs_services.py # ElevenLabs TTS generation
├── image_services.py      # Multi-provider image generation (Together AI, OpenAI, Gemini)
├── video_editor.py        # V1 video assembly with MoviePy
├── v2_editor.py           # V2 video assembly (word-timestamps + visual beats)
├── v3_editor.py           # V3 video assembly (focus-aware zoom)
├── v5_editor.py           # V5 video assembly (uploaded video clips)
├── v6_editor.py           # V6 hybrid assembly (image + video scenes)
├── subtitle_renderer.py   # Subtitle rendering engine
├── script_content.py      # Default scene content
├── main.py                # CLI entry point (standalone usage)
├── requirements.txt       # Python dependencies
├── frontend/              # React frontend (Vite)
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   ├── App.css        # Application styles
│   │   ├── main.jsx       # React entry point
│   │   └── index.css      # Global styles
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **npm**
- **FFmpeg** (required for video processing)
- API key(s):
  - **ElevenLabs API key** (default speech provider — ultra-realistic TTS)
  - **Together AI API key** (default image provider — FLUX models)
  - **Google Gemini API key** (optional — for Gemini TTS and image generation)
  - **OpenAI API key** (optional — for OpenAI TTS and DALL·E / GPT Image)

### 1. Clone the repository

```bash
git clone https://github.com/samiabat/personal.git
cd personal
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here  # Default speech provider
TOGETHER_API_KEY=your_together_ai_api_key_here   # Default image provider
GEMINI_API_KEY=your_gemini_api_key_here          # Optional
OPENAI_API_KEY=your_openai_api_key_here          # Optional
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 5. Run the application

```bash
python app.py
```

The app will start at **http://localhost:8000**

## 💻 Development Mode

For development with hot-reload on both frontend and backend:

**Terminal 1 — Backend:**
```bash
uvicorn app:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

The frontend dev server runs on `http://localhost:5173` with API requests proxied to the backend.

## 📖 Usage Guide

### Generate a Video

1. Open the app in your browser at `http://localhost:8000`
2. Configure your preferred **speech provider**, **model**, **voice**, and **image model** in the Settings panel
3. Set the desired **resolution** and **aspect ratio**
4. Go to the **Generate Video** tab
5. Paste your scenes JSON or click **Load Default Scenes**
6. Click **🚀 Generate Video**
7. Watch the progress bar as scenes are generated
8. Download the completed video when ready

### Scene JSON Format

Scenes are defined as a JSON array. Each scene has two fields:

```json
[
  {
    "voiceover": "The text that will be spoken as voiceover for this scene.",
    "prompt": "A detailed description of the image to generate for this scene."
  },
  {
    "voiceover": "Another scene's voiceover text goes here.",
    "prompt": "Another scene's image prompt goes here."
  }
]
```

### Test Audio & Image

Use the **🧪 Test Audio / Image** tab to verify your settings:

- **Test Audio**: Enter sample text and generate audio with your chosen speech model and voice
- **Test Image**: Enter a prompt and generate an image with your chosen image model

This helps verify that your API keys and model selections are working correctly before starting a full video generation.

## 🛠️ API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/generate-video` | Start a video generation job |
| `GET` | `/api/progress/{job_id}` | SSE stream of generation progress |
| `GET` | `/api/status/{job_id}` | Get current job status |
| `GET` | `/api/download/{job_id}` | Download the generated video |
| `POST` | `/api/test-audio` | Test audio generation |
| `POST` | `/api/test-image` | Test image generation |
| `GET` | `/api/models` | Get available models and options |
| `GET` | `/api/default-scenes` | Get the built-in default scenes |

## 🎛️ Available Models

### Speech Models

| Provider | Model | Description |
|----------|-------|-------------|
| **ElevenLabs** | `eleven_multilingual_v2` | Multilingual v2 — **default** |
| ElevenLabs | `eleven_turbo_v2_5` | Turbo v2.5 (fast) |
| ElevenLabs | `eleven_flash_v2_5` | Flash v2.5 (fastest) |
| ElevenLabs | `eleven_turbo_v2` | Turbo v2 |
| Google | `gemini-2.5-pro-preview-tts` | Gemini 2.5 Pro TTS |
| Google | `gemini-2.5-flash-preview-tts` | Gemini 2.5 Flash TTS |
| OpenAI | `tts-1` | OpenAI TTS-1 |
| OpenAI | `tts-1-hd` | OpenAI TTS-1 HD |
| OpenAI | `gpt-4o-mini-tts` | GPT-4o Mini TTS |

> 💡 You can also type a custom model name for any provider.

### Image Models

| Provider | Model | Description |
|----------|-------|-------------|
| **Together AI** | `black-forest-labs/FLUX.1-schnell` | FLUX.1 Schnell — **default** |
| Together AI | `black-forest-labs/FLUX.1-dev` | FLUX.1 Dev |
| Together AI | `black-forest-labs/FLUX.1.1-pro` | FLUX.1.1 Pro |
| Together AI | `stabilityai/stable-diffusion-xl-base-1.0` | Stable Diffusion XL |
| Together AI | `Lykon/dreamshaper-xl-v2-turbo` | DreamShaper XL v2 Turbo |
| Gemini | `gemini-3.1-flash-image-preview` | Gemini 3.1 Flash Image Preview |
| Gemini | `gemini-2.0-flash-preview-image-generation` | Gemini 2.0 Flash Image Gen |
| Gemini | `imagen-3.0-generate-002` | Imagen 3.0 |

> 💡 Custom model names are supported via the UI checkbox.

## 🖥️ CLI Usage

The original CLI pipeline is still available:

```bash
python main.py
```

This uses the default scenes from `script_content.py` and generates `assets/final_trading_short.mp4`.

## 📋 Requirements

- `google-genai` — Google Gemini API client
- `pillow` — Image handling
- `moviepy` — Video editing and assembly
- `python-dotenv` — Environment variable loading
- `fastapi` — Web framework
- `uvicorn[standard]` — ASGI server
- `python-multipart` — Form data parsing
- `openai` — OpenAI API client
- `sse-starlette` — Server-Sent Events for FastAPI

## 📄 License

This project is for personal use.