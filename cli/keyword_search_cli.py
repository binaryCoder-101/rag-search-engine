
import argparse
import json
import string
from lib.search_utils import tokenizer, has_matching_token

def main() -> None:
    json_file = open('data/movies.json') 
    json_file_obj = json.load(json_file)


    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    result = []

    for movie in json_file_obj["movies"]:
        translator = str.maketrans('', '', string.punctuation)

        query = tokenizer(args.query)
        movie_title = tokenizer(movie["title"])

        if has_matching_token(query, movie_title) and len(result) < 5:
            result.append(movie)

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            for movie in result:
                print(f"- {movie['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()