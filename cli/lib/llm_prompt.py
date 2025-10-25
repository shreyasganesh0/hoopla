import os
from dotenv import load_dotenv
from google import genai

MODEL = "gemini-2.0-flash-001"

class Llm:

    def __init__(self, model = MODEL):

        self.model = model
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")

        self.client = genai.Client(api_key=api_key)


    def enhance_prompt(self, query, mode):

        print(f"Using key {api_key[:6]}...")

        sys_prompt = ""
        match(mode):

            case "spell":
                sys_prompt = f"""Fix any spelling errors in this movie search query.

                            Only correct obvious typos. Don't change correctly spelled words.

                            Query: "{query}"

                            If no errors, return the original query.
                            Corrected:"""
            case "rewrite":

                sys_prompt = f"""Rewrite this movie search query to be more specific and searchable.

                            Original: "{query}"

                            Consider:
                            - Common movie knowledge (famous actors, popular films)
                            - Genre conventions (horror = scary, animation = cartoon)
                            - Keep it concise (under 10 words)
                            - It should be a google style search query that's very specific
                            - Don't use boolean logic

                            Examples:

                            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                            Rewritten query:"""
            case "expand":
                
                sys_prompt = f"""Expand this movie search query with related terms.

                            Add synonyms and related concepts that might appear in movie descriptions.
                            Keep expansions relevant and focused.
                            This will be appended to the original query.

                            Examples:

                            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                            - "action movie with bear" -> "action thriller bear chase fight adventure"
                            - "comedy with bear" -> "comedy funny bear humor lighthearted"

                            Query: "{query}"
                            """

        model_obj = self.client.models.generate_content(model = self.model, contents = sys_prompt) 

        return model_obj.text

    def rerank_prompt(self, doc, query, mode):

        sys_prompt = ""
        rank_list = []

        match(mode):

            case "individual":

                sys_prompt = f"""Rate how well this movie matches the search query.

                            Query: "{query}"
                            Movie: {doc.get("title", "")} - {doc.get("document", "")}

                            Consider:
                            - Direct relevance to query
                            - User intent (what they're looking for)
                            - Content appropriateness

                            Rate 0-10 (10 = perfect match).
                            Give me ONLY the number in your response, no other text or explanation.

                            Score:"""
            case "batch":
                sys_prompt = f"""Rank these movies by relevance to the search query.

                            Query: "{query}"

                            Movies:
                            {doc}

                            Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

                            [75, 12, 34, 2, 1]
                            """

        model_obj = self.client.models.generate_content(model = self.model, contents = sys_prompt) 

        return model_obj.text

