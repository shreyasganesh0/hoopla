import pickle
import math
import os
import string
from collections import Counter

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:

    def __init__(self, stopwords, stemmer):

        self.index = {}
        self.docmap = {}
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.N = 0

    def __add_document(self, doc_id, text):

        token_list = self.tokenizer(text)

        for token in token_list:

            self.index.setdefault(token, set()).add(doc_id) # add doc to doc_id set
            self.term_frequencies.setdefault(doc_id, Counter())[token] += 1
            val = self.doc_lengths.get(doc_id, 0.0) 
            self.doc_lengths[doc_id] = val + 1.0

    def tokenizer(self, my_str):

        t_table = str.maketrans(dict.fromkeys(string.punctuation, None))

        return [
                self.stemmer.stem(word) 
                for word in my_str.lower().translate(t_table).split()
                if word and word not in self.stopwords
                ]

    def get_tf(self, doc_id: int, term: str) -> int:

        token = self.tokenizer(term)
        if len(token) > 1: raise Exception("too many tokens in term")

        token = token[0]

        return self.term_frequencies.get(doc_id, Counter()).get(token, 0)

    def get_idf(self, term: str):

        doc_count = self.N
        term_doc_count = len(self.get_documents(term))
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:

        df = len(self.get_documents(term))

        bm25idf = math.log(1 + ((self.N - df + 0.5) / (df + 0.5)))

        return bm25idf
    
    def __get_avg_doc_length(self) -> float:

        sum = 0.0

        for doc_len in self.doc_lengths.values():

            sum += doc_len

        return sum/float(len(self.doc_lengths.values()))


    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:

        tf = self.get_tf(doc_id, term)

        length_norm = 1 - b + b * (self.doc_lengths.get(doc_id, 0.0) / self.__get_avg_doc_length())

        bm25tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return bm25tf

    def get_documents(self, term):

        token = self.tokenizer(term)

        if len(token) > 1: raise exception("too many tokens in terms")

        term = token[0]

        doc_ids = sorted(list(self.index.get(term, [])))

        ret_list = []
        for doc_id in doc_ids:

            ret_list.append(self.docmap.get(doc_id, {}))
        return ret_list

    def bm25(self, doc_id, term):

        bm25_idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term, BM25_K1, BM25_B)

        return bm25_tf * bm25_idf
    
    def bm25_search(self, query, limit):

        query_tokens = self.tokenizer(query)
        if len(query_tokens) == 0: raise Exception("too few tokens")

        scores = {}

        for token in query_tokens:

            curr_score = 0

            for doc_id in self.index.get(token, []):

                curr_score = self.bm25(doc_id, token)

                val = scores.get(doc_id, 0.0)
                scores[doc_id] = val + curr_score

        score_sort = sorted(scores.items(), key = lambda a: a[1], reverse = True)

        i = 1
        for curr_doc_id, curr_doc_score in score_sort[:limit]:

            curr_doc = self.docmap[curr_doc_id]

            i += 1
        return score_sort[:limit]


    
    def build(self, movies):

        for movie in movies:

            inp_text = f"{movie["title"]} {movie["description"]}"

            self.__add_document(movie["id"], inp_text)
            self.docmap[movie["id"]] = movie
            self.N += 1

    def save(self):

        try:
            os.mkdir("cache")
        except FileExistsError:
            print("dir cache exists")

        with open("cache/index.pkl", "wb") as f_idx:
            pickle.dump(self.index, f_idx)
        with open("cache/docmap.pkl", "wb")as f_docmap:
            pickle.dump(self.docmap, f_docmap)
        with open("cache/term_frequencies.pkl", "wb") as f_term_freq:
            pickle.dump(self.term_frequencies, f_term_freq)
        with open("cache/doc_lengths.pkl", "wb") as f_doc_lengths:
            pickle.dump(self.doc_lengths, f_doc_lengths)

    def load(self):

        try:
            with open("cache/index.pkl", "rb") as f_idx:
                self.index = pickle.load(f_idx)
            with open("cache/docmap.pkl", "rb") as f_docmap:
                self.docmap = pickle.load(f_docmap)
            with open("cache/term_frequencies.pkl", "rb") as f_term_freq:
                self.term_frequencies = pickle.load(f_term_freq)
            with open("cache/doc_lengths.pkl", "rb") as f_doc_lengths:
                self.doc_lengths = pickle.load(f_doc_lengths)

            self.N = len(self.docmap.keys())

        except Exception as e:

            raise Exception(e)

        

