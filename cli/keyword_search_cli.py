
import argparse
from lib.search_utils import load_movies
from lib.keyword_search import tokenizer, InvertedIndex, has_matching_token

def main() -> None: 

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Creates inverse index and stores in pkl files")

    args = parser.parse_args()

    match args.command:
        case "search":
            movies = load_movies()
            results = []
            for movie in movies:
                query = tokenizer(args.query)
                movie_title = tokenizer(movie["title"])
                if has_matching_token(query, movie_title) and len(results) < 5:
                    results.append(movie)
            print(f"Searching for: {args.query}")
            for movie in results:
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