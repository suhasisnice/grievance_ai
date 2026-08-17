from dotenv import load_dotenv
import os
load_dotenv()

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_name = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
print(f"Using model name: '{model_name}'")

try:
    response = client.models.generate_content(model=model_name, contents="Say hello")
    print("SUCCESS:", response.text)
except Exception as e:
    print(f"FULL ERROR: {e}")