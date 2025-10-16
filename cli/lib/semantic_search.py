from sentence_transformers import SentenceTransformer
import string
import json
import os
import numpy as np

class SemanticSearch:

    def __init__(self):

        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embeddings(self, text):

        if not text.strip(string.whitespace):

            raise ValueError("String was empty or contained only whitespace")
        embedding = self.model.encode([text])
        
        return embedding[0]
    
    def build_embeddings(self, documents):

        print("building")

        doc_list = []

        for v in documents:

            doc_list.append(f"{v["title"]}: {v["description"]}")

        self.embeddings = self.model.encode(doc_list, show_progress_bar = True)
        
        with open("cache/movie_embeddings.py", "wb") as f_embed:

            np.save(f_embed, self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents):

        self.documents = documents

        for doc in documents:

            self.document_map[doc["id"]] = doc

        if os.path.exists("cache/movie_embeddings.py"):

            print("loading embeddings from file")
            self.embeddings = np.load("cache/movie_embeddings.py")

        if len(self.embeddings) == len(documents):

            return self.embeddings
        else:

            return self.build_embeddings(documents)

def embed_query_text(query):

    sem_search = SemanticSearch()

    embedding = sem_search.generate_embeddings(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():

    sem_search = SemanticSearch()

    documents = []
    with open("data/movies.json", "r") as f_movies:
        data = json.load(f_movies)
        documents = data["movies"]
    embeddings = sem_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_text(text):

    sem_search = SemanticSearch()

    embedding = sem_search.generate_embeddings(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():

    sem_search = SemanticSearch()

    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")
