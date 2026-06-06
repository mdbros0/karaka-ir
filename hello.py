import os
from dotenv import load_dotenv
from google import genai

load_dotenv()  # must run before os.getenv — populates os.environ from .env

api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your_key_here":
    raise SystemExit(
        "GEMINI_API_KEY not set. Open .env and replace 'your_key_here' with your real key."
    )

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello and confirm you're working.",
)

print(response.text)
