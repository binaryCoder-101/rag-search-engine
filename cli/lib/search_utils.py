import json
from typing import Any, TypedDict

SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP_SIZE = 0
MAX_SIZE_SEMANTIC_CHUNK = 4
SCORE_PRECISION = 3
ALPHA = 0.5
RRF_K = 60

class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]

def load_movies() -> list[dict]:
    json_file = open('data/movies.json') 
    return json.load(json_file)["movies"]

def load_stopwords() -> list[str]:
     with open('data/stopwords.txt', 'r') as f:
         return f.read().splitlines()
     
def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
