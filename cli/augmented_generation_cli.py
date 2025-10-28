import argparse
from lib.augmented_generation import perform_rag, perform_summary, perform_rag_with_citations, perform_question_answering

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Search and summarize results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to summarize"
    )
    citations_parser = subparsers.add_parser(
        "citations", help="Search and generate answer with citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query for citation answer")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to use for citations"
    )

    question_parser = subparsers.add_parser(
    "question", help="Search and answer a question conversationally"
    )
    question_parser.add_argument("question", type=str, help="Question to ask based on search results")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to provide context"
    )
    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            print(f"Performing RAG for query: '{query}'")
            search_results, response, error = perform_rag(query)

            if error:
                print(f"\nError: {error}")
                if search_results:
                    print("\nSearch Results (obtained before error):")
                    for result in search_results:
                        print(f"  - {result.get('title', 'N/A')}")
                return

            print("\nSearch Results:")
            for result in search_results:
                print(f"  - {result.get('title', 'N/A')}")

            print("\nRAG Response:")
            print(response)

        case "summarize":
            query = args.query
            limit = args.limit
            print(f"Performing search and summarize for query: '{query}' (limit={limit})")
            search_results, summary, error = perform_summary(query, limit)

            if error:
                print(f"\nError: {error}")
                if search_results:
                    print("\nSearch Results (obtained before error):")
                    for result in search_results:
                        print(f"  - {result.get('title', 'N/A')}")
                return

            print("\nSearch Results:")
            for result in search_results:
                print(f"  - {result.get('title', 'N/A')}")

            print("\nLLM Summary:")
            print(summary)

        case "citations":
            query = args.query
            limit = args.limit
            print(f"Performing search and citation generation for query: '{query}' (limit={limit})")
            search_results, answer, error = perform_rag_with_citations(query, limit)

            if error:
                print(f"\nError: {error}")
                if search_results:
                    print("\nSearch Results (obtained before error):")
                    for result in search_results:
                        print(f"  - {result.get('title', 'N/A')}")
                return

            print("\nSearch Results:")
            for i, result in enumerate(search_results, 1):
                print(f"  [{i}] {result.get('title', 'N/A')}") 

            print("\nLLM Answer:")
            print(answer)

        case "question":
            question = args.question
            limit = args.limit
            print(f"Performing search to answer question: '{question}' (limit={limit})")
            search_results, answer, error = perform_question_answering(question, limit)

            if error:
                print(f"\nError: {error}")
                if search_results:
                    print("\nSearch Results (obtained before error):")
                    for result in search_results:
                        print(f"  - {result.get('title', 'N/A')}")
                return

            print("\nSearch Results:")
            for result in search_results:
                print(f"  - {result.get('title', 'N/A')}")

            print("\nAnswer:")
            print(answer)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
