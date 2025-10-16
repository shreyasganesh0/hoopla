#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("verify", help="Verify existance of model")
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for text")
    embed_parser.add_argument("text", type=str, help="Word to find embedding of")

    args = parser.parse_args()

    match args.command:

        case "verify":

            verify_model()

        case "embed_text":

            embed_text(args.text)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
