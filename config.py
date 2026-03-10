import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def get_gemini_client():
    from google import genai
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    return genai.Client(api_key=GEMINI_API_KEY)

def get_openai_client():
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return OpenAI(api_key=OPENAI_API_KEY)