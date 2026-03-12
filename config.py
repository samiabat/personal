import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from the .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Initialize the global client to be used across the app
client = genai.Client(api_key=API_KEY)