import json

SEARCH_LIMIT = 5

def load_movies() -> list[dict]:
    json_file = open('data/movies.json') 
    return json.load(json_file)["movies"]

def load_stopwords() -> list[str]:
     with open('data/stopwords.txt', 'r') as f:
         return f.read().splitlines()