#!/usr/bin/env python3

import argparse
import json
import string


def main() -> None:


    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            fp = open("data/movies.json", "r")
            data = json.load(fp)
            movies_list = []

            t_table = str.maketrans(dict.fromkeys(string.punctuation, None))

            for movie in data["movies"]:

                if args.query.lower().translate(t_table) in movie["title"].lower().translate(t_table):

                    movies_list.append(movie)
            movies_list.sort(key = lambda a : a["id"])

            i = 1
            for mv in movies_list:
                print(f"{i}. {mv["title"]}") 
                i +=1

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
