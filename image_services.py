import base64
import io
from PIL import Image
from config import gemini_client, openai_client, together_client

# ---------- Gemini ----------
def generate_image_gemini(prompt: str, output_path: str, model: str = "gemini-3.1-flash-image-preview"):
    """Generate an image using Google Gemini."""
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY not configured")
    from google.genai import types

    img_response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
                image_size="512",
            ),
        ),
    )
    for part in img_response.parts:
        if part.inline_data:
            img = part.as_image()
            img.save(output_path)
            return True
    return False


# ---------- OpenAI ----------
def generate_image_openai(prompt: str, output_path: str, model: str = "gpt-image-1"):
    """Generate an image using OpenAI image models."""
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY not configured")

    result = openai_client.images.generate(
        model=model,
        prompt=prompt,
        size="1536x1024",
        n=1,
    )

    image_data = base64.b64decode(result.data[0].b64_json)
    img = Image.open(io.BytesIO(image_data))
    img.save(output_path)
    return True


# ---------- Together AI ----------
TOGETHER_MODELS = {
    "flux.1-schnell": "black-forest-labs/FLUX.1-schnell",
    "dreamshaper": "Lykon/dreamshaper-xl-v2-turbo",
    "flux.1-dev": "black-forest-labs/FLUX.1-dev",
    "stable-diffusion-xl": "stabilityai/stable-diffusion-xl-base-1.0",
}

TOGETHER_SIZES = {
    "1024x576": {"width": 1024, "height": 576},
    "1280x720": {"width": 1280, "height": 720},
}

def generate_image_together(
    prompt: str,
    output_path: str,
    model: str = "flux.1-schnell",
    size: str = "1024x576",
):
    """Generate an image using Together AI. No 'steps' parameter is used."""
    if not together_client:
        raise RuntimeError("TOGETHER_API_KEY not configured")

    model_id = TOGETHER_MODELS.get(model, model)
    dims = TOGETHER_SIZES.get(size, TOGETHER_SIZES["1024x576"])

    response = together_client.images.generate(
        prompt=prompt,
        model=model_id,
        width=dims["width"],
        height=dims["height"],
        n=1,
    )

    img_data = base64.b64decode(response.data[0].b64_json)
    img = Image.open(io.BytesIO(img_data))
    img.save(output_path)
    return True


# ---------- Dispatcher ----------
def generate_image(
    prompt: str,
    output_path: str,
    provider: str = "gemini",
    model: str = None,
    together_size: str = "1024x576",
):
    """Route image generation to the chosen provider."""
    if provider == "gemini":
        return generate_image_gemini(prompt, output_path, model or "gemini-3.1-flash-image-preview")
    elif provider == "openai":
        return generate_image_openai(prompt, output_path, model or "gpt-image-1")
    elif provider == "together":
        return generate_image_together(prompt, output_path, model or "flux.1-schnell", together_size)
    else:
        raise ValueError(f"Unknown image provider: {provider}")
