import os
from dotenv import load_dotenv
from google import genai

def llm_prompt(query):

    MODEL = "gemini-2.0-flash-001"
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"Using key {api_key[:6]}...")

    client = genai.Client(api_key=api_key)
    model_obj = client.models.generate_content(model = MODEL, contents = f"""Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.
    Corrected:""")

    return model_obj.text
