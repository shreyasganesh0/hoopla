import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Generate embedding for an image and show its shape"
    )
    verify_parser.add_argument("image_path", type=str, help="Path to the image file")

    search_parser = subparsers.add_parser(
        "image_search", help="Search for movies using an image query"
    )
    search_parser.add_argument("image_path", type=str, help="Path to the image file for searching")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)
            if results:
                print("\nSearch Results:")
                for i, res in enumerate(results, 1):
                    description_snippet = (res.get("description", "") or "").split('\n')[0][:100] + "..."
                    print(f"{i}. {res.get('title', 'N/A')} (similarity: {res.get('score', 0.0):.3f})")
                    print(f"   {description_snippet}")
            else:
                print("No results found or an error occurred during search.")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
