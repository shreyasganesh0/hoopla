import argparse
import json
from lib.hybrid_search import weighted_search, rrf_search
from lib.llm_prompt import Llm


def main():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Weighted hybrid search"
    )
    weighted_parser.add_argument("query", type=str, help="The search query")
    weighted_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 vs. semantic (0.0 to 1.0)",
    )

    rrf_parser = subparsers.add_parser("rrf-search", help="RRF hybrid search")
    rrf_parser.add_argument("query", type=str, help="The search query")
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    rrf_parser.add_argument(
        "--k", type=int, default=60, help="K parameter for RRF (default: 60)"
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        default="",
        choices=["", "spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        default="",
        choices=["", "individual", "batch", "cross_encoder"],
        help="Reranking method",
    )
    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the search results with an LLM",
    )

    args = parser.parse_args()

    if args.command == "weighted-search":
        results = weighted_search(args.query, args.alpha, args.limit)
        print(f"Weighted search results for: {args.query} (alpha={args.alpha})\n")
        for i, curr_res in enumerate(results, 1):
            doc_snippet = curr_res["document"].split("\n")[0][:100]
            print(f"{i}. {curr_res['title']}")
            print(f"   Hybrid Score: {curr_res['hybrid_score']:.3f}")
            print(
                f"   BM25: {curr_res['bm25_norm']:.3f}, Semantic: {curr_res['sem_norm']:.3f}"
            )
            print(f"   {doc_snippet}...")

    if args.command == "rrf-search":
        results = rrf_search(
            args.query,
            args.k,
            args.limit,
            args.enhance,
            args.rerank_method,
        )

        print(f"RRF search results for: {args.query}\n")
        for i, curr_res in enumerate(results, 1):
            doc_snippet = curr_res["document"].split("\n")[0][:100]
            print(f"{i}. {curr_res['title']}")

            if args.rerank_method == "individual" or args.rerank_method == "batch":
                print(f"   LLM Rank: {curr_res.get('llm_rank', 'N/A')}")
            elif args.rerank_method == "cross_encoder":
                print(
                    f"   Cross-Encoder Score: {curr_res.get('cross_encoder_score', -1):.3f}"
                )
            else:
                print(f"   RRF Score: {curr_res.get('rrf_score', 0.0):.3f}")
            print(f"   {doc_snippet}...")

        if args.evaluate:
            print("\n--- Evaluation Report ---")

            formatted_results_for_llm = []
            for res in results:
                snippet = res.get("document", "").split("\n")[0][:100]
                formatted_results_for_llm.append(
                    f"{res.get('title', 'N/A')}: {snippet}..."
                )

            try:
                llm = Llm()
                eval_response = llm.evaluate_prompt(
                    args.query, formatted_results_for_llm
                )

                scores = json.loads(eval_response)

                if len(scores) != len(results):
                    print(
                        f"\nError: LLM returned {len(scores)} scores for {len(results)} results."
                    )
                    return

                for i, (res, score) in enumerate(zip(results, scores), 1):
                    print(f"{i}. {res['title']}: {score}/3")

            except json.JSONDecodeError:
                print(f"\nError: Could not parse LLM evaluation response.")
                print(f"Raw response: {eval_response}")
            except Exception as e:
                print(f"\nAn error occurred during evaluation: {e}")

    elif args.command == "normalize":
        print(f"Original: {args.scores}")
        print(f"Normalized: {normalize(args.scores)}")

    elif not args.command in ["weighted-search", "rrf-search", "normalize"]:
        parser.print_help()


if __name__ == "__main__":
    main()
