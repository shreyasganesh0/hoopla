import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity

class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc.get('title', '')}: {doc.get('description', '')}" for doc in documents]
        print("Generating text embeddings for multimodal search...")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        print(f"Generated {len(self.text_embeddings)} text embeddings.")

    def embed_image(self, image_path):
        try:
            img = Image.open(image_path)
            embedding = self.model.encode([img], show_progress_bar=False)
            return embedding[0]
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def search_with_image(self, image_path, limit=5):
        image_embedding = self.embed_image(image_path)
        if image_embedding is None:
            return []

        scores = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            scores.append({
                "id": self.documents[i].get("id"),
                "title": self.documents[i].get("title"),
                "description": self.documents[i].get("description"),
                "score": similarity
            })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:limit]

def verify_image_embedding(image_path):
    searcher = MultimodalSearch(documents=[]) 
    embedding = searcher.embed_image(image_path)
    if embedding is not None:
        print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    movies = []
    try:
        with open("data/movies.json", "r") as f:
            data = json.load(f)
            movies = data["movies"]
    except FileNotFoundError:
        print("Error: data/movies.json not found.")
        return []
    except json.JSONDecodeError:
        print("Error: Could not decode data/movies.json.")
        return []

    searcher = MultimodalSearch(documents=movies)
    results = searcher.search_with_image(image_path)
    return results
