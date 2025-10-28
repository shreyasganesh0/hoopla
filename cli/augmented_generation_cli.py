import argparse
from lib.augmented_generation import perform_rag

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            print(f"Performing RAG for query: '{query}'")

            search_results, rag_response, error = perform_rag(query)

            if error:
                print(f"\nError: {error}")
                # Still print search results if they were found before the LLM error
                if search_results:
                    print("\nSearch Results (obtained before error):")
                    for result in search_results:
                        print(f"  - {result.get('title', 'N/A')}")
                return # Exit if there was an error

            print("\nSearch Results:")
            for result in search_results:
                print(f"  - {result.get('title', 'N/A')}")

            print("\nRAG Response:")
            print(rag_response)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
