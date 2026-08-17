from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print(f"API key loaded: {api_key[:10] if api_key else 'NONE'}...")

try:
    from google import genai
    print("✓ google.genai imported")
    
    client = genai.Client(api_key=api_key)
    print("✓ Client created")
    
    models = list(client.models.list())
    print(f"✓ Found {len(models)} models")
    
    for m in models:
        print(f"  - {m.name}")
        
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")