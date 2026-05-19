
import argparse
import json
from lib.search_utils import tokenizer, has_matching_token
from lib.search_utils import load_movies
from lib.keyword_search import InvertedIndex

def main() -> None: 
    json_file_obj = load_movies()

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Creates inverse index and stores in pkl file")

    args = parser.parse_args()

    match args.command:
        case "search":
            result = []
            for movie in json_file_obj:
                query = tokenizer(args.query)
                movie_title = tokenizer(movie["title"])
                if has_matching_token(query, movie_title) and len(result) < 5:
                    result.append(movie)
            print(f"Searching for: {args.query}")
            for movie in result:
                print(f"- {movie['title']}")
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
            print(f"{inverted_index.get_documents('merida')[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()