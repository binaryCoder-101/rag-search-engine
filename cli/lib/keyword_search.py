from pickle import dump
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


def search_command(query: str, limit: int = SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokenizer(query)
        title_tokens = tokenizer(movie["title"])
        if has_matching_token(query_tokens, title_tokens) and len(results) < limit:
            results.append(movie)

    return results


def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    docs = inverted_index.get_documents("merida")
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