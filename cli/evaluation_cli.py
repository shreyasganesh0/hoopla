import argparse

from lib.evaluation import evaluvate_scores


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    evaluvate_scores(limit)


if __name__ == "__main__":
    main()
