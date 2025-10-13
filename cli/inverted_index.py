import pickle
import os
import string

def tokenizer(my_str, stopwords, stemmer):


    t_table = str.maketrans(dict.fromkeys(string.punctuation, None))

    return [
            stemmer.stem(word) 
            for word in my_str.lower().translate(t_table).split()
            if word and word not in stopwords
            ]

class InvertedIndex:

    def __init__(self, stopwords, stemmer):

        self.index = {}
        self.docmap = {}
        self.stopwords = stopwords
        self.stemmer = stemmer

    def __add_document(self, doc_id, text):

        token_list = tokenizer(text, self.stopwords, self.stemmer)

        for token in token_list:

            self.index.setdefault(token, set()).add(doc_id) # add doc to doc_id set

    def get_documents(self, term):

        term = term.lower()

        return sorted(list(self.index.get(term, [])))
    
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

        f_idx = open("cache/index.pkl", "wb")
        f_docmap = open("cache/docmap.pkl", "wb")

        pickle.dump(self.index, f_idx)
        pickle.dump(self.docmap, f_docmap)

        

