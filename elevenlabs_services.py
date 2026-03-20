import struct
from config import get_elevenlabs_client


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 22050, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Wrap raw PCM data with a WAV file header."""
    data_size = len(pcm_data)
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ",
        16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return header + pcm_data


def generate_audio_elevenlabs(
    text: str,
    output_path: str,
    model: str = "eleven_multilingual_v2",
    voice: str = "3TStB8f3X3To0Uj5R7RK",
    api_key: str = "",
) -> bool:
    """Generates voiceover using ElevenLabs TTS and saves as WAV to output_path."""
    client = get_elevenlabs_client(api_key)
    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id=voice,
        model_id=model,
        output_format="pcm_22050",
    )
    pcm_data = b"".join(audio_generator)
    if pcm_data:
        wav_bytes = _pcm_to_wav(pcm_data, sample_rate=22050)
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        return True
    return False
