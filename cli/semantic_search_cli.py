import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command
from lib.search_utils import SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify model by printing it's information")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for given text")
    embed_text_parser.add_argument("text", type=str, help="Text whose embedding is to be generated")

    subparsers.add_parser("verify_embeddings", help="verify movie data embedding by printing it's information")
    
    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embedding for given text")
    embed_query_parser.add_argument("query", type=str, help="Text whose embedding is to be generated")

    search_parser = subparsers.add_parser("search", help="Semantic search command")
    search_parser.add_argument("query", type=str, help="query")
    search_parser.add_argument("--limit", type=int, nargs='?', default=SEARCH_LIMIT, help="Search result limit")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)

if __name__ == "__main__":
    main()