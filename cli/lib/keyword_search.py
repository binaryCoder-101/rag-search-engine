from pickle import dump, load
import string
from nltk.stem import PorterStemmer
from .search_utils import SEARCH_LIMIT, load_movies, load_stopwords

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, input_text):
        input_tokens = tokenizer(input_text)
        for token in input_tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        doc_ids = list(self.index[term])
        doc_ids.sort()
        return doc_ids
    
    def build(self):
        movies_json = load_movies()

        for movie in movies_json:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self):
        with open("cache/index.pkl", "wb") as file:
            dump(self.index, file)
        with open("cache/docmap.pkl", "wb") as file:
            dump(self.docmap, file)

    def load(self):
        with open("cache/index.pkl", "rb") as file:
            self.index = load(file)
        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = load(file)

def search_command(query: str, limit: int = SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    
    try:
        idx.load()
    except FileNotFoundError:
        print("Pickle file doesn't exist")
        return

    query_tokens = tokenizer(query)
    results = []
    for token in query_tokens:
        matching_ids = idx.index.get(token, [])
        for doc_id in matching_ids:
            results.append(doc_id)
            if len(results) >= 5:
                break
        if len(results) >= 5:
            break
        

    return results


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    """Checks if at least one token from the query matches any part of a token from the title."""
    
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def tokenizer(text: str) -> list[str]:
    """Preprocess a string and return a list of clean tokens."""

    translator = str.maketrans('', '', string.punctuation)
    tokens = text.translate(translator).lower().split()

    stop_words = load_stopwords()
    
    stemmer = PorterStemmer()
    for index, token in enumerate(tokens):
        if token in stop_words:
            tokens.remove(token)
            continue
        tokens[index] = stemmer.stem(token)

    return tokens