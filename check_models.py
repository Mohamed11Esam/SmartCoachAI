"""Check available Gemini models"""
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key: {api_key[:10]}...{api_key[-4:]}")

client = genai.Client(api_key=api_key)

print("\nListing available models...")
try:
    for model in client.models.list():
        print(f"  - {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")

print("\nTrying direct generation...")
try:
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents="Say hello"
    )
    print(f"Success! Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
