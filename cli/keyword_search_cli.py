#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer


def main() -> None:


    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    

    match args.command:

        case "search":

            print(f"Searching for: {args.query}")

            f_movies = open("data/movies.json", "r")
            data = json.load(f_movies)

            f_stopwords = open("data/stopwords.txt", "r")
            stopwords = f_stopwords.read().splitlines()

            stemmer = PorterStemmer()

            movies_list = []
            limit = 5

            t_table = str.maketrans(dict.fromkeys(string.punctuation, None))
            cleaner = lambda my_str:[stemmer.stem(word) 
                                     for word in my_str.lower().translate(t_table).split()
                                     if word and word not in stopwords]

            query_list = cleaner(args.query)

            movie_count = 0

            for movie in data["movies"]:

                title_list = cleaner(movie["title"])

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
