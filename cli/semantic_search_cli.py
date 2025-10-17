#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, verify_embeddings, embed_text, embed_query_text, search_query, chunk

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("verify", help="Verify existance of model")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", 
                                                     help="Verify embeddings of documents")

    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for text")
    embed_parser.add_argument("text", type=str, help="Word to find embedding of")

    search_parser = subparsers.add_parser("search", help="Generate embedding for text")
    search_parser.add_argument("query", type=str, help="Query to find embedding of")
    search_parser.add_argument("--limit", type=int, 
                              help="Optional limit to number of matches (default 5)", default=5)

    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for query")
    embed_query_parser.add_argument("query", type=str, help="Query to find embedding of")

    chunk_parser = subparsers.add_parser("chunk", help="Split documents into chunks before encoding")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, help="Chunk size to split documents on", default=200)
    args = parser.parse_args()

    match args.command:

        case "verify":

            verify_model()

        case "verify_embeddings":

            verify_embeddings()

        case "embed_text":

            embed_text(args.text)

        case "embedquery":

            embed_query_text(args.query)

        case "search":

            search_query(args.query, args.limit)

        case "chunk":

            chunk(args.text, args.chunk_size)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
