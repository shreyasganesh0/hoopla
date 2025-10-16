from sentence_transformers import SentenceTransformer
import string

class SemanticSearch:

    def __init__(self):

        self.model = SentenceTransformer('all-MiniLM-L6-v2') 

    def generate_embeddings(self, text):

        if not text.strip(string.whitespace):

            raise ValueError("String was empty or contained only whitespace")
        embedding = self.model.encode([text])
        
        return embedding[0]
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
