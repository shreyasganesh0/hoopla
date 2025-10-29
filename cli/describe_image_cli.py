import argparse
from lib.describe_image import describe_image_and_rewrite_query

def main():
    parser = argparse.ArgumentParser(description="Rewrite search query based on image context.")
    parser.add_argument("--image", required=True, type=str, help="Path to the image file")
    parser.add_argument("--query", required=True, type=str, help="Text query to rewrite based on the image")

    args = parser.parse_args()

    rewritten_query, usage_metadata, error = describe_image_and_rewrite_query(args.image, args.query)

    if error:
        print(f"\nError: {error}")
        return

    print(f"Rewritten query: {rewritten_query}")

    if usage_metadata is not None:
        print(f"Total tokens:    {usage_metadata.total_token_count}")
    else:
        if "Error" not in rewritten_query:
             print("Usage metadata not available.")

if __name__ == "__main__":
    main()
