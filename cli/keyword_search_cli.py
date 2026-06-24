
import argparse
from lib.keyword_search import build_command, document_content_command, search_command, tf_command, idf_command, tfidf_command, bm25_idf_command, bm25_tf_command, bm25_search_command
from lib.search_utils import BM25_K1, BM25_B, SEARCH_LIMIT

def main() -> None: 

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Creates inverse index and stores in pkl files")

    tf_parser = subparsers.add_parser("tf", help="Give term frequency for a term in a document with given ID")
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Term whose frequency is to be searched")

    idf_parser = subparsers.add_parser("idf", help="Calculate IDF value for a given term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF score for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF value for a given term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document id")
    tfidf_parser.add_argument("term", type=str, help="Term whose TF-IDF value is to be calculated")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=SEARCH_LIMIT, help="Search result limit")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency for term {args.term} in document {args.doc_id}: {tf}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")
        case "bm25search":
            bm25_search = bm25_search_command(args.query, args.limit)
            for count, movie_id in enumerate(bm25_search, start=1):
                movie_details = document_content_command(movie_id)
                print(f"{count}. ({movie_id}) {movie_details['title']} - Score: {bm25_search[movie_id]:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()