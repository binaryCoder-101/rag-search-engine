import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command, chunk_command ,semantic_chunk_command, embed_chunks_command, search_chunked_command
from lib.search_utils import SEARCH_LIMIT, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_SIZE, MAX_SIZE_SEMANTIC_CHUNK

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
    
    chunk_parser = subparsers.add_parser("chunk", help="Chunk command")
    chunk_parser.add_argument("text", type=str, help="query")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=DEFAULT_CHUNK_SIZE, help="Search result limit")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=DEFAULT_OVERLAP_SIZE, help="Size of chunking overlap")
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic chunk command")
    semantic_chunk_parser.add_argument("text", type=str, help="query")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=MAX_SIZE_SEMANTIC_CHUNK, help="Search result limit")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=DEFAULT_OVERLAP_SIZE, help="Size of semantic chunking overlap")

    subparsers.add_parser("embed_chunks", help="Create embeddings for document chunks")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Semantic chunked search command")
    search_chunked_parser.add_argument("query", type=str, help="query")
    search_chunked_parser.add_argument("--limit", type=int, nargs='?', default=SEARCH_LIMIT, help="Search result limit")

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
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            search_chunked_command(args.query, args.limit)
            
if __name__ == "__main__":
    main()