import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize the Gemini client (only if key is available)
gemini_client = None
if GEMINI_API_KEY:
    from google import genai
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize the OpenAI client (only if key is available)
openai_client = None
if OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Together AI uses requests directly
together_client = None
if TOGETHER_API_KEY:
    import together
    together_client = together.Together(api_key=TOGETHER_API_KEY)

# Backward compatibility alias
client = gemini_client