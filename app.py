"""Flask web frontend for the video generation pipeline."""
import os
import threading
import json
from flask import Flask, render_template, request, jsonify
from main import main as run_pipeline, ASSET_DIR, setup_directories
from script_content import scenes
from image_services import TOGETHER_MODELS

app = Flask(__name__)

# Simple in-memory state for progress tracking
pipeline_state = {
    "running": False,
    "progress": 0,
    "total": len(scenes),
    "message": "",
    "error": None,
}


@app.route("/")
def index():
    """Render the main configuration page."""
    available_keys = {
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "together": bool(os.getenv("TOGETHER_API_KEY")),
    }
    return render_template(
        "index.html",
        scenes=scenes,
        available_keys=available_keys,
        together_models=list(TOGETHER_MODELS.keys()),
    )


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Start the video pipeline in a background thread."""
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline is already running"}), 409

    data = request.get_json(force=True)
    image_provider = data.get("image_provider", "gemini")
    image_model = data.get("image_model") or None
    together_size = data.get("together_size", "1024x576")
    enable_shake = data.get("enable_shake", False)
    resume = data.get("resume", True)

    pipeline_state.update(running=True, progress=0, message="Starting…", error=None)

    def _run():
        try:
            run_pipeline(
                image_provider=image_provider,
                image_model=image_model,
                together_size=together_size,
                enable_shake=enable_shake,
                resume=resume,
            )
            pipeline_state["message"] = "Pipeline finished successfully!"
        except Exception as exc:
            pipeline_state["error"] = str(exc)
            pipeline_state["message"] = f"Error: {exc}"
        finally:
            pipeline_state["running"] = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/status")
def api_status():
    """Return current pipeline progress."""
    # Compute progress by counting existing assets
    setup_directories()
    generated = 0
    for i in range(len(scenes)):
        audio_ok = os.path.exists(f"{ASSET_DIR}/scene_{i}_audio.wav")
        image_ok = os.path.exists(f"{ASSET_DIR}/scene_{i}_image.jpg")
        if audio_ok and image_ok:
            generated += 1
    pipeline_state["progress"] = generated
    return jsonify(pipeline_state)


@app.route("/api/scenes")
def api_scenes():
    """Return scene details with generation status."""
    setup_directories()
    result = []
    for i, scene in enumerate(scenes):
        result.append({
            "index": i,
            "voiceover": scene["voiceover"][:120] + "…" if len(scene["voiceover"]) > 120 else scene["voiceover"],
            "audio_exists": os.path.exists(f"{ASSET_DIR}/scene_{i}_audio.wav"),
            "image_exists": os.path.exists(f"{ASSET_DIR}/scene_{i}_image.jpg"),
        })
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
