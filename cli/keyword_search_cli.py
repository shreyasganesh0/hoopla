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

    tf_parser = subparsers.add_parser("tf", help="Get term frequencies for given term")
    tf_parser.add_argument("doc_id", type=int, help="document where term exists")
    tf_parser.add_argument("term", type=str, help="Token to search for")

    args = parser.parse_args()
    
    f_movies = open("data/movies.json", "r")
    data = json.load(f_movies)
    movies = data["movies"]
    f_movies.close()

    f_stopwords = open("data/stopwords.txt", "r")
    stopwords = f_stopwords.read().splitlines()
    f_stopwords.close()
    
    stemmer = PorterStemmer()
    inv_idx = inverted_index.InvertedIndex(stopwords, stemmer)

    match args.command:

        case "search":

            print(f"Searching for: {args.query}")

            movies_list = []
            limit = 5

            query_list = inv_idx.tokenizer(args.query)

            movie_count = 0

            for query_token in query_list:


                try:
                    inv_idx.load()
                except Exception as e:
                    print(e)
                    return

                docs = inv_idx.get_documents(query_token)

                docs_count = 0
                docs_list = []
                for doc in docs:

                    docs_list.append(f"title: {doc["title"]}, id:{doc["id"]}")
                    docs_count += 1

                    if (docs_count >= limit):
                        i = 1

                        for docs in docs_list:
                            print(f"{i}. {docs}") 
                            i +=1

                        break

        case "build":


            inv_idx.build(movies)
            inv_idx.save()

        case "tf":


            try:
                inv_idx.load()
                print(f"term frequency of {args.term} in {args.doc_id} is {inv_idx.get_tf(args.doc_id, args.term)}")
            except Exception as e:
                    print(e)
                    return

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
