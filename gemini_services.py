import struct
from google.genai import types
from config import client

def _parse_audio_mime_type(mime_type: str) -> dict:
    bits_per_sample, rate = 16, 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try: rate = int(param.split("=", 1)[1])
            except: pass
        elif param.startswith("audio/L"):
            try: bits_per_sample = int(param.split("L", 1)[1])
            except: pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}

def _convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    params = _parse_audio_mime_type(mime_type)
    bps, rate, channels = params["bits_per_sample"], params["rate"], 1
    data_size = len(audio_data)
    block_align = channels * (bps // 8)
    byte_rate = rate * block_align
    
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ",
        16, 1, channels, rate, byte_rate, block_align, bps, b"data", data_size         
    )
    return header + audio_data

def generate_audio(text: str, output_path: str):
    """Generates Charon voiceover and saves to output_path."""
    tts_config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
            )
        )
    )
    
    response_stream = client.models.generate_content_stream(
        model="gemini-2.5-pro-preview-tts",
        contents=text,
        config=tts_config,
    )

    all_audio_data = bytearray()
    mime_type = "audio/L16;rate=24000"
    
    for chunk in response_stream:
        if chunk.parts and chunk.parts[0].inline_data:
            inline_data = chunk.parts[0].inline_data
            all_audio_data.extend(inline_data.data)
            if inline_data.mime_type:
                mime_type = inline_data.mime_type

    if all_audio_data:
        final_wav_bytes = _convert_to_wav(bytes(all_audio_data), mime_type)
        with open(output_path, "wb") as f:
            f.write(final_wav_bytes)
        return True
    return False

def generate_image(prompt: str, output_path: str):
    """Generates a 16:9 image and saves to output_path."""
    img_response = client.models.generate_content(
        model='gemini-3.1-flash-image-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio="16:9", 
                image_size="512"
            ),
        ),
    )
    
    for part in img_response.parts:
        if part.inline_data:
            img = part.as_image()
            img.save(output_path)
            return True
    return False