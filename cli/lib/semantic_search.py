from sentence_transformers import SentenceTransformer
import string
import json
import os
import re
import numpy as np

class SemanticSearch:

    def __init__(self, model_name):

        self.model = SentenceTransformer(model_name) 
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
        
        with open("cache/movie_embeddings.npy", "wb") as f_embed:

            np.save(f_embed, self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents):

        self.documents = documents

        for doc in documents:

            self.document_map[doc["id"]] = doc

        if os.path.exists("cache/movie_embeddings.npy"):

            print("loading embeddings from file")
            self.embeddings = np.load("cache/movie_embeddings.npy")

        if len(self.embeddings) == len(documents):

            return self.embeddings
        else:

            return self.build_embeddings(documents)

    def search(self, query, limit):

        if len(self.embeddings) == 0:

            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embeddings(query)

        scores_list = []

        for i in range(len(self.embeddings)):

            curr_score = cosine_similarity(query_embedding, self.embeddings[i])
            scores_list.append((curr_score, self.documents[i]))

        scores_list.sort(key = lambda a: a[0], reverse = True)

        ret_list = []

        for (score, doc) in scores_list[:limit]:

            ret_list.append({"score": score, "title": doc["title"], "description": doc["description"]})

        return ret_list

class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):

        print("building")

        chunk_list = []
        metadata_list = []

        for movie_idx, v in enumerate(documents):
            if not v.get("description") or not v["description"].strip():
                continue

            text = v["description"]
            
            chunks = sem_chunk(text, 4, 1)

            chunk_list.extend(chunks)

            for chunk_idx_in_doc, _ in enumerate(chunks):
                metadata_list.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx_in_doc,
                    "total_chunks": len(chunks)
                })
        self.chunk_embeddings = self.model.encode(chunk_list, show_progress_bar = True)
        self.chunk_metadata = metadata_list
        
        with open("cache/chunk_embeddings.npy", "wb") as f_embed:

            np.save(f_embed, self.chunk_embeddings)

        with  open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunk_list)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:

        self.documents = documents

        for doc in documents:

            self.document_map[doc["id"]] = doc

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):

            print("loading embeddings from file")
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]

            return self.chunk_embeddings
        else:

            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):

        query_embedding = super().generate_embeddings(query)

        chunk_score_list = []
        chunk_idx = 0
        
        for embedding in self.chunk_embeddings:

            sim = cosine_similarity(embedding, query_embedding)
            movie_idx = self.chunk_metadata[chunk_idx]["movie_idx"]
            chunk_score_list.append(
                                    {"chunk_idx": chunk_idx, 
                                     "movie_idx": movie_idx, 
                                     "score": sim
                                    }
                                   )
            chunk_idx += 1

        movie_score_dict = {}
        for chunk_score in chunk_score_list:

            curr_movie_idx = chunk_score["movie_idx"]
            curr_score = chunk_score["score"]

            if curr_movie_idx not in movie_score_dict or movie_score_dict[curr_movie_idx] < curr_score:
                movie_score_dict[curr_movie_idx] = curr_score 

        sort_movies = sorted(movie_score_dict.items(), key = lambda a: a[1], reverse = True)

        results = []

        for movie_idx, score in sort_movies:
            curr_movie = self.documents[movie_idx]
            res = format_search_result(curr_movie["id"], curr_movie["title"], curr_movie["description"][:100], score)
            results.append(res)

        return results[:limit]

def search_chunked(query, limit):

    movies = []
    with open("data/movies.json", "r") as f:
        data = json.load(f)
        movies = data["movies"]
    sem_search = ChunkedSemanticSearch()

    embeddings = sem_search.load_or_create_chunk_embeddings(movies)

    results = sem_search.search_chunks(query, limit)
    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document']}...")


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: any
) -> dict[str, any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, 3),
        "metadata": metadata if metadata else {},
    }

def embed_chunks():

    movies = []
    with open("data/movies.json", "r") as f:
        data = json.load(f)
        movies = data["movies"]
    sem_search = chunkedsemanticsearch()
    
    embeddings = sem_search.load_or_create_chunk_embeddings(movies)

    print(f"Generated {len(embeddings)} chunked embeddings")


def chunk(text, chunk_size, overlap):

    pattern = f'[{re.escape(string.whitespace)}]'
    text_list = [s for s in re.split(pattern, text) if s]

    print(f"Chunking {len(text)} characters")

    x = 1
    i = 0
    chunk_list = []
    while (i < len(text_list)):

        curr_word_list = text_list[i:i + chunk_size]

        curr_print = f"{x}." 
        for word in curr_word_list:
            curr_print += " " + word
        print(curr_print)
        x += 1
        i += (chunk_size - overlap)

def sem_chunk(text, max_chunk_size, overlap):
    text = text.strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    n_sentences = len(sentences)

    if n_sentences == 1 and not sentences[0].endswith(('.', '?', '!')):
        return sentences

    chunks = []
    i = 0
    while i < n_sentences: 
        
        current_slice = sentences[i : i + max_chunk_size]

        chunk_sentences = [s.strip() for s in current_slice if s.strip()]

        if chunk_sentences:
            chunks.append(" ".join(chunk_sentences))
        
        i += max_chunk_size - overlap

    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")       
    return chunks

def search_query(query, limit):

    sem_search = SemanticSearch()

    documents = []
    with open("data/movies.json", "r") as f_movies:
        data = json.load(f_movies)
        documents = data["movies"]
    embeddings = sem_search.load_or_create_embeddings(documents)
    res = sem_search.search(query, limit)

    i = 1 
    for d in res:

        print(f"{i}. {d["title"]}: ({d["score"]})\n   {d["description"]}\n")
        i += 1


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

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
