
import argparse
from lib.keyword_search import build_command, search_command, tf_command, idf_command, tfidf_command, bm25_idf_command

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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()