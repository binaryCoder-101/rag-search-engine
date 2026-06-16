from pickle import dump, load
import string
from nltk.stem import PorterStemmer
from .search_utils import SEARCH_LIMIT, load_movies, load_stopwords
from collections import Counter, defaultdict

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id, input_text):
        input_tokens = tokenizer(input_text)
        token_count = Counter()
        for token in input_tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            token_count[token] += 1
        self.term_frequencies[doc_id] = token_count
        print(doc_id, token_count)

    def get_documents(self, term):
        doc_ids = list(self.index.get(term, set()))
        doc_ids.sort()
        return doc_ids
    
    def get_tf(self, doc_id, term) -> int:
        return self.term_frequencies[doc_id][term]

    def build(self):
        movies_json = load_movies()
        for movie in movies_json:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, doc_description)
            self.docmap[movie["id"]] = movie

    def save(self):
        with open("cache/index.pkl", "wb") as file:
            dump(self.index, file)
        with open("cache/docmap.pkl", "wb") as file:
            dump(self.docmap, file)
        with open("cache/term_frequencies.pkl", "wb") as file:
            dump(self.term_frequencies, file)

    def load(self):
        with open("cache/index.pkl", "rb") as file:
            self.index = load(file)
        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = load(file)
        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = load(file)

def search_command(query: str, limit: int = SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    
    try:
        idx.load()
    except FileNotFoundError:
        print("Pickle file doesn't exist")
        return

    query_tokens = tokenizer(query)
    result_doc_id = []
    for token in query_tokens:
        matching_ids = idx.get_documents(token)
        for doc_id in matching_ids:
            result_doc_id.append(doc_id)
            if len(result_doc_id) >= limit:
                break
        if len(result_doc_id) >= limit:
            break

    result_movies = []
    for doc_id in result_doc_id:
        result_movies.append(idx.docmap.get(doc_id))

    return result_movies

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()

    tokenized_term = single_term_tokenizer(term)
    tf = idx.get_tf(doc_id, tokenized_term)

    return tf

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    print(idx.get_documents("brave"))

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

def single_term_tokenizer(term: str) -> str:
    result = tokenizer(term)
    if len(result) != 1:
        raise ValueError("Token length should be one")
    return result[0]