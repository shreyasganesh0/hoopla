# lib/hybrid_search.py

import os
import json
from nltk.stem import PorterStemmer
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        f_stopwords = open("data/stopwords.txt", "r")
        stopwords = f_stopwords.read().splitlines()
        f_stopwords.close()
        
        stemmer = PorterStemmer()
        self.idx = InvertedIndex(stopwords, stemmer)
        if not os.path.exists("cache/index.pkl"):
            moives = []
            f_movies = open("data/movies.json", "r")
            data = json.load(f_movies)
            movies = data["movies"]
            f_movies.close()
            self.idx.build(movies)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, int(limit))

    def weighted_search(self, query, alpha, limit=5):

        bm25_res = self._bm25_search(query, limit) # List of (doc_id, score)
        sem_res = self.semantic_search.search_chunks(query, limit) # List of dicts

        combined_results = {}
        
        for doc_id, score in bm25_res:
            if doc_id not in self.idx.docmap: continue
            doc = self.idx.docmap[doc_id]
            combined_results[doc_id] = {
                "id": doc_id,
                "title": doc["title"],
                "document": doc["description"],
                "bm25_score": score,
                "sem_score": 0.0,
            }

        for res in sem_res:
            doc_id = res["id"]
            if doc_id in combined_results:
                combined_results[doc_id]["sem_score"] = res["score"]
            else:
                combined_results[doc_id] = {
                    "id": doc_id,
                    "title": res["title"],
                    "document": self.idx.docmap.get(doc_id, {}).get("description", res["document"]),
                    "bm25_score": 0.0,
                    "sem_score": res["score"]
                }

        bm25_scores = [res["bm25_score"] for res in combined_results.values()]
        sem_scores = [res["sem_score"] for res in combined_results.values()]

        norm_bm25 = normalize(bm25_scores)
        norm_sem = normalize(sem_scores)
        
        results_list = []
        i = 0
        for doc_id, data in combined_results.items():
            data["bm25_norm"] = norm_bm25[i]
            data["sem_norm"] = norm_sem[i]
            
            data["hybrid_score"] =  hybrid_score(data["bm25_norm"], data["sem_norm"], alpha)
            
            results_list.append(data)
            i += 1
        results_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results_list


    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize(scores):
    if not scores:
        return []
        
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for score in scores:
        normalized_scores.append((score - min_score) / (max_score - min_score))
    
    return normalized_scores

def weighted_search(query, alpha, limit):
    movies = []
    with open("data/movies.json", "r") as f:
        data = json.load(f)
        movies = data["movies"]
    hy_search = HybridSearch(movies)

    large_limit = limit * 500 
    res = hy_search.weighted_search(query, alpha, large_limit)

    for i, curr_res in enumerate(res[:limit], 1):
        doc_snippet = curr_res["document"].split("\n")[0][:100]
        
        print(f"{i}. {curr_res['title']}")
        print(f"   Hybrid Score: {curr_res['hybrid_score']:.3f}")
        print(f"   BM25: {curr_res['bm25_norm']:.3f}, Semantic: {curr_res['sem_norm']:.3f}")
        print(f"   {doc_snippet}...")
