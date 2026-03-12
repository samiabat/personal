import base64
import requests
from config import get_openai_client, get_together_client, TOGETHER_API_KEY


def generate_image_openai(prompt: str, output_path: str, model: str = "gpt-image-1", size: str = "1024x1024"):
    """Generates an image using OpenAI and saves to output_path."""
    client = get_openai_client()
    result = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        n=1,
    )
    # The response can contain b64_json or url depending on model
    image_data = result.data[0]
    if hasattr(image_data, "b64_json") and image_data.b64_json:
        img_bytes = base64.b64decode(image_data.b64_json)
        with open(output_path, "wb") as f:
            f.write(img_bytes)
    elif hasattr(image_data, "url") and image_data.url:
        img_resp = requests.get(image_data.url, timeout=120)
        img_resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(img_resp.content)
    else:
        raise ValueError("No image data returned from OpenAI")
    return True


def generate_image_togetherai(prompt: str, output_path: str, model: str = "black-forest-labs/FLUX.1-schnell",
                               width: int = 1024, height: int = 576):
    """Generates an image using Together AI and saves to output_path."""
    client = get_together_client()
    response = client.images.generate(
        prompt=prompt,
        model=model,
        width=width,
        height=height,
        n=1,
    )
    # Together AI returns b64_json in the response
    image_data = response.data[0]
    if hasattr(image_data, "b64_json") and image_data.b64_json:
        img_bytes = base64.b64decode(image_data.b64_json)
        with open(output_path, "wb") as f:
            f.write(img_bytes)
    elif hasattr(image_data, "url") and image_data.url:
        img_resp = requests.get(image_data.url, timeout=120)
        img_resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(img_resp.content)
    else:
        raise ValueError("No image data returned from Together AI")
    return True
