from pickle import dump, load
import string
from nltk.stem import PorterStemmer
from .search_utils import SEARCH_LIMIT, BM25_K1, BM25_B, load_movies, load_stopwords
from collections import Counter, defaultdict
import math

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}

    def __add_document(self, doc_id, input_text):
        input_tokens = tokenizer(input_text)
        token_count = Counter()
        for token in input_tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            token_count[token] += 1
        self.term_frequencies[doc_id] = token_count
        self.doc_lengths[doc_id] = len(input_tokens)

    def get_documents(self, term):
        doc_ids = list(self.index.get(term, set()))
        doc_ids.sort()
        return doc_ids
    
    def get_tf(self, doc_id, term) -> int:
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[term])
        
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)        #Total number of documents in the corpus
        df = len(self.index[term])  #Document frequency (number of documents containing term)
        
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1:int=BM25_K1, b:int=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return bm25_tf

    def __get_avg_doc_length(self) -> float:
        num_docs = len(self.doc_lengths)
        if num_docs == 0:
            return 0.0
        sum = 0
        for doc_id in self.doc_lengths:
            sum += self.doc_lengths[doc_id]

        return sum/num_docs

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
        with open("cache/doc_lengths.pkl", "wb") as file:
            dump(self.doc_lengths, file)

    def load(self):
        with open("cache/index.pkl", "rb") as file:
            self.index = load(file)
        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = load(file)
        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = load(file)
        with open("cache/doc_lengths.pkl", "rb") as file:
            self.doc_lengths = load(file)

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

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    tokenized_term = single_term_tokenizer(term)
    idf = idx.get_idf(tokenized_term)

    return idf

def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    tokenized_term = single_term_tokenizer(term)
    tf = idx.get_tf(doc_id, tokenized_term)
    idf = idx.get_idf(tokenized_term)

    return tf * idf

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    tokenized_term = single_term_tokenizer(term)
    bm25_idf = idx.get_bm25_idf(tokenized_term)

    return bm25_idf

def bm25_tf_command(doc_id: int, term: str, k1:int=BM25_K1) -> float:
    idx = InvertedIndex()
    idx.load()

    tokenized_term = single_term_tokenizer(term)
    bm25_tf = idx.get_bm25_tf(doc_id, tokenized_term, k1)

    return bm25_tf

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    """Checks if at least one token from the query matches any part of a token from the title."""
    
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenizer(text: str) -> list[str]:
    """Preprocess a string and return a list of clean tokens."""

    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)

    filtered_words = []
    stop_words = load_stopwords()
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)

    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))

    return stemmed_words

def single_term_tokenizer(term: str) -> str:
    result = tokenizer(term)
    if len(result) != 1:
        raise ValueError("Token length should be one")
    return result[0]