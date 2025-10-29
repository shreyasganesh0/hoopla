import mimetypes
import os
# Use genai directly assuming 'types' is accessible via the main import
from google import genai
from .llm_prompt import Llm

def describe_image_and_rewrite_query(image_path, text_query):
    rewritten_query = "Error: Could not generate rewritten query."
    usage_metadata = None
    error_message = None

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    if not os.path.exists(image_path):
        error_message = f"Error: Image file not found at {image_path}"
        return rewritten_query, usage_metadata, error_message

    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
    except Exception as e:
        error_message = f"Error reading image file: {e}"
        return rewritten_query, usage_metadata, error_message

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    try:
        image_part = genai.types.Part.from_bytes(data=img_data, mime_type=mime)

        parts = [
            system_prompt,
            image_part,
            text_query.strip(),
        ]
    except AttributeError:
         error_message = "Error: Could not access 'genai.types.Part'. Check library structure/version."
         return rewritten_query, usage_metadata, error_message
    except Exception as e:
         error_message = f"Error creating image part: {e}"
         return rewritten_query, usage_metadata, error_message


    try:
        llm = Llm()
        response = llm.rewrite_query_with_image(parts)

        # Safely access text attribute from the response object
        response_text = getattr(response, 'text', None)

        if response_text:
             if "Error generating response" in response_text:
                 error_message = response_text 
                 rewritten_query = "Failed to generate rewritten query due to API error."
             else:
                 rewritten_query = response_text.strip()
        elif hasattr(response, 'parts') and response.parts:
             rewritten_query = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
             if not rewritten_query: 
                 error_message = "Error: LLM response parts contained no text."
        else:
            error_message = "Error: Could not extract rewritten query from LLM response."


        if hasattr(response, 'usage_metadata'):
            usage_metadata = response.usage_metadata

    except Exception as e:
        error_message = f"Error during LLM call: {e}"
        # Ensure rewritten_query reflects the failure state
        rewritten_query = "Failed due to exception during LLM call."


    return rewritten_query, usage_metadata, error_message
