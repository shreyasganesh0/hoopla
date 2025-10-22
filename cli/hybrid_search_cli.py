import argparse

from lib.hybrid_search import normalize, weighted_search, rrf_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize list of scores")
    normalize_parser.add_argument("scores", type=float, nargs='+', help="List of float scores to normalize", default=[])

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Normalize list of scores")
    weighted_search_parser.add_argument("query", type=str, help="Query to searhc")
    weighted_search_parser.add_argument("--alpha", type=float, help="Alpha value for wts", default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, help="Limit for number of queries", default=5)
    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform RRF hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="Query to search")
    rrf_search_parser.add_argument("--k", type=int, help="k value for RRF score", default=60)
    rrf_search_parser.add_argument("--limit", type=int, help="Limit for number of results", default=5)
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.scores)

        case "weighted-search":

            weighted_search(args.query, args.alpha, args.limit)

        case "rrf-search":

            rrf_search(args.query, args.k, args.limit, args.enhance)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
