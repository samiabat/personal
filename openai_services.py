from config import get_openai_client

def generate_audio_openai(text: str, output_path: str, model: str = "tts-1", voice: str = "alloy", api_key: str = ""):
    """Generates voiceover using OpenAI TTS and saves to output_path."""
    client = get_openai_client(api_key)
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format="wav"
    )
    response.stream_to_file(output_path)
    return True
