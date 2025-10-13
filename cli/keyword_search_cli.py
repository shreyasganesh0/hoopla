#!/usr/bin/env python3

import argparse
import json
from nltk.stem import PorterStemmer

import inverted_index

def main() -> None:


    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index and save it to disk")

    args = parser.parse_args()
    
    f_movies = open("data/movies.json", "r")
    data = json.load(f_movies)
    movies = data["movies"]
    f_movies.close()

    f_stopwords = open("data/stopwords.txt", "r")
    stopwords = f_stopwords.read().splitlines()
    f_stopwords.close()
    
    stemmer = PorterStemmer()

    match args.command:

        case "search":

            print(f"Searching for: {args.query}")



            movies_list = []
            limit = 5


            query_list = inverted_index.tokenizer(args.query, stopwords, stemmer)

            movie_count = 0

            for movie in movies: 

                title_list = inverted_index.tokenizer(movie["title"], stopwords, stemmer)

                if any(query_word in title_word for query_word in query_list for title_word in title_list):
                    movies_list.append(movie)
                    movie_count += 1

                if (movie_count >= limit):

                    break

            movies_list.sort(key = lambda a : a["id"])

            i = 1

            for mv in movies_list:
                print(f"{i}. {mv["title"]}") 
                i +=1

        case "build":

            inv_idx = inverted_index.InvertedIndex(stopwords, stemmer)

            inv_idx.build(movies)
            inv_idx.save()

            print(f"First document for token 'merida' = {inv_idx.get_documents("merida")[0]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
