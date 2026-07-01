from sentence_transformers import SentenceTransformer
import numpy as np
import os
from .search_utils import load_movies, SEARCH_LIMIT, DEFAULT_CHUNK_SIZE

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("The provided text is empty")
        
        embeddings = self.model.encode([text])

        return embeddings[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        movie_strings = []
        for document in documents:
            self.document_map[f"{document['id']}"] = document
            movie_strings.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[f"{document['id']}"] = document
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if self.embeddings.shape[0] == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        similarity_scores = []

        for i, movie_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, movie_embedding)
            similarity_scores.append((similarity_score, self.documents[i]))

        similarity_scores.sort(key=lambda x: x[0], reverse=True)

        results = []

        for movie_score_data in similarity_scores[:limit]:
            data = {"score": movie_score_data[0],
                    "title": movie_score_data[1]['title'],
                    "description": movie_score_data[1]['description']
                    }
            
            results.append(data)

        return results

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query: str, limit: int=SEARCH_LIMIT) -> None:
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)

    search_results = ss.search(query, limit)

    for i, result in enumerate(search_results):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"{result['description'][:100]}...")
        print("\n")

def chunk_command(text: str, size: int=DEFAULT_CHUNK_SIZE):
    words = text.split(" ")
    print(f"Chunking {len(text)} characters")
    chunks = []
    current = []
    for word in words:
        current.append(word)
        if len(current) == size:
            chunks.append(" ".join(current))
            current = []
    # don't forget the leftovers!
    if current:
        chunks.append(" ".join(current))
    
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")
    # for word in split_text:
    #     sentence = ""
    #     word_count = 0
    #     line_count = 0
    #     if word_count <= size:
    #         sentence += " " + word
    #         word_count += 1
    #     line_count += 1
    #     print(f"{line_count}. {sentence}")