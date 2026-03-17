import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

def get_gemini_client(api_key: str = ""):
    from google import genai
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY not provided. Add it in Settings or set it in .env file.")
    return genai.Client(api_key=key)

def get_openai_client(api_key: str = ""):
    from openai import OpenAI
    key = api_key or OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY not provided. Add it in Settings or set it in .env file.")
    return OpenAI(api_key=key)

def get_together_client(api_key: str = ""):
    from together import Together
    key = api_key or TOGETHER_API_KEY
    if not key:
        raise ValueError("TOGETHER_API_KEY not provided. Add it in Settings or set it in .env file.")
    return Together(api_key=key)

def get_elevenlabs_client(api_key: str = ""):
    from elevenlabs.client import ElevenLabs
    key = api_key or ELEVENLABS_API_KEY
    if not key:
        raise ValueError("ELEVENLABS_API_KEY not provided. Add it in Settings or set it in .env file.")
    return ElevenLabs(api_key=key)