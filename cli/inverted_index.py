import pickle
import math
import os
import string
from collections import Counter


class InvertedIndex:

    def __init__(self, stopwords, stemmer):

        self.index = {}
        self.docmap = {}
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.term_frequencies = {}

    def __add_document(self, doc_id, text):

        token_list = self.tokenizer(text)

        for token in token_list:

            self.index.setdefault(token, set()).add(doc_id) # add doc to doc_id set
            self.term_frequencies.setdefault(doc_id, Counter())[token] += 1

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

        doc_count = len(self.docmap.keys())
        term_doc_count = len(self.get_documents(term))
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_documents(self, term):

        token = self.tokenizer(term)

        if len(token) > 1: raise exception("too many tokens in terms")

        term = token[0]

        doc_ids = sorted(list(self.index.get(term, [])))

        ret_list = []
        for doc_id in doc_ids:

            ret_list.append(self.docmap.get(doc_id, {}))
        return ret_list
    
    def build(self, movies):

        for movie in movies:

            inp_text = f"{movie["title"]} {movie["description"]}"

            self.__add_document(movie["id"], inp_text)
            self.docmap[movie["id"]] = movie

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

    def load(self):

        try:
            with open("cache/index.pkl", "rb") as f_idx:
                self.index = pickle.load(f_idx)
            with open("cache/docmap.pkl", "rb") as f_docmap:
                self.docmap = pickle.load(f_docmap)
            with open("cache/term_frequencies.pkl", "rb") as f_term_freq:
                self.term_frequencies = pickle.load(f_term_freq)

        except Exception as e:

            raise Exception(e)

        

