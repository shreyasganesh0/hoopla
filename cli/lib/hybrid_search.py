# lib/hybrid_search.py

import time
import os
import json
from nltk.stem import PorterStemmer
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .llm_prompt import Llm
from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

        bm25_res = self._bm25_search(query, limit) 
        sem_res = self.semantic_search.search_chunks(query, limit) 

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

        fetch_limit = limit 
        bm25_res = self._bm25_search(query, fetch_limit) 
        sem_res = self.semantic_search.search_chunks(query, fetch_limit)

        combined_results = {}

        for rank, (doc_id, _) in enumerate(bm25_res, 1): # Rank starts at 1
            if doc_id not in self.idx.docmap: continue
            doc = self.idx.docmap[doc_id]
            score = rrf_score(rank, k)
            
            combined_results[doc_id] = {
                "id": doc_id,
                "title": doc["title"],
                "document": doc["description"],
                "bm25_rank": rank,
                "sem_rank": 0, 
                "rrf_score": score
            }

        for rank, res in enumerate(sem_res, 1): 
            doc_id = res["id"]
            score = rrf_score(rank, k)

            if doc_id in combined_results:
                combined_results[doc_id]["rrf_score"] += score
                combined_results[doc_id]["sem_rank"] = rank
            else:
                if doc_id not in self.idx.docmap: continue
                doc = self.idx.docmap[doc_id]
                combined_results[doc_id] = {
                    "id": doc_id,
                    "title": doc["title"],
                    "document": doc["description"],
                    "bm25_rank": 0,
                    "sem_rank": rank,
                    "rrf_score": score
                }

        results_list = sorted(combined_results.values(), key=lambda x: x["rrf_score"], reverse=True)
        
        return results_list

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

def rrf_score(rank, k=60):
    return 1.0 / (k + rank)


def rrf_search(query, k, limit, enhance, rerank):
    movies = []
    with open("data/movies.json", "r") as f:
        data = json.load(f)
        movies = data["movies"]
    hy_search = HybridSearch(movies)

    llm = Llm()
    if enhance:
        enhanced_query = llm.enhance_prompt(query, enhance)
        logging.debug(f"Enhanced query: {enhanced_query}")
        query = enhanced_query 
    
    fetch_limit = limit * 5 

    res = hy_search.rrf_search(query, k, fetch_limit)
    logging.debug(f"RRF results (pre-rerank): {[doc['title'] for doc in res]}")

    if rerank:
        
        if rerank == "individual":
            for i, curr_res in enumerate(res):
                doc = curr_res["document"]
                try:
                    res[i]["llm_rank"] = float(llm.rerank_prompt(curr_res, query, rerank))
                except (ValueError, TypeError):
                    res[i]["llm_rank"] = 0.0 
                time.sleep(1) 
            
            res.sort(key = lambda a: a.get("llm_rank", 0.0), reverse=True)

        if rerank == "batch":
            docs_to_rerank = []
            for curr_res in res: 
                docs_to_rerank.append({
                    "id": curr_res["id"],
                    "title": curr_res["title"],
                    "document": curr_res["document"]
                })
            
            doc_list_str = json.dumps(docs_to_rerank, indent=2)

            json_resp = llm.rerank_prompt(doc_list_str, query, rerank)
            
            try:
                ranked_ids = json.loads(json_resp)
                
                rank_map = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, 1)}

                for curr_res in res:
                    curr_res["llm_rank"] = rank_map.get(curr_res["id"], 99999) 

                res.sort(key = lambda a: a["llm_rank"])

            except (json.JSONDecodeError, TypeError):
                for i, curr_res in enumerate(res):
                    curr_res["llm_rank"] = i + 1

    if rerank == "cross_encoder":
            
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

            pairs = []
            for doc in res:
                doc_string = f"{doc.get('title', '')} - {doc.get('document', '')}"
                pairs.append([query, doc_string])

            scores = cross_encoder.predict(pairs)

            for i, curr_res in enumerate(res):
                curr_res["cross_encoder_score"] = scores[i]
            
            res.sort(key=lambda x: x.get('cross_encoder_score', -float('inf')), reverse=True)
    final_results = res[:limit]
    logging.debug(f"Final results (post-rerank): {[doc['title'] for doc in final_results]}")
    return final_results 
