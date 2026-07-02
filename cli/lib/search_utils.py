import json

SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP_SIZE = 0

def load_movies() -> list[dict]:
    json_file = open('data/movies.json') 
    return json.load(json_file)["movies"]

def load_stopwords() -> list[str]:
     with open('data/stopwords.txt', 'r') as f:
         return f.read().splitlines()