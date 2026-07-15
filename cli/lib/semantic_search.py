from sentence_transformers import SentenceTransformer
import numpy as np
import os
from .search_utils import load_movies, SEARCH_LIMIT, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_SIZE, MAX_SIZE_SEMANTIC_CHUNK, format_search_result
import re
import json

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
         super().__init__(model_name)
         self.chunk_embeddings = None
         self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        all_chunks = []
        chunk_metadata = []

        for i, document in enumerate(documents):
            current_document_chunks = []
            self.document_map[f"{document['id']}"] = document
            if not document['description']:
                continue
            current_document_chunks = semantic_chunk_command(document['description'],4,1)
            for j, chunk in enumerate(current_document_chunks):
                all_chunks.append(chunk)
                metadata = {
                    "movie_idx": i,
                    "chunk_idx": j,
                    "total_chunks": len(current_document_chunks)
                }
                chunk_metadata.append(metadata)
        
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[f"{document['id']}"] = document
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                 data = json.load(f)

            metadata_list = data["chunks"]
            self.chunk_metadata = metadata_list
            return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_chunk_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        chunk_scores = []
        best_chunk_score = {}

        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            chunk_score = cosine_similarity(query_embedding, chunk_embedding)
            movie_idx = self.chunk_metadata[i]["movie_idx"]

            if movie_idx not in best_chunk_score or chunk_score > best_chunk_score[movie_idx]:
                best_chunk_score[movie_idx] = chunk_score

            movie_score_dict = {
                "chunk_idx": i,
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": chunk_score
            }
            chunk_scores.append(movie_score_dict)

        sorted_best_chunk_score = sorted(best_chunk_score.items(), key=lambda item: item[1], reverse=True)

        results = []

        for movie_idx, score in sorted_best_chunk_score[:limit]:
            doc = self.documents[movie_idx]
           
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"][:100],
                score=score
            )
            
            results.append(formatted_result)

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

def chunk_command(text: str, size: int=DEFAULT_CHUNK_SIZE, overlap: int=DEFAULT_OVERLAP_SIZE):
    words = text.split(" ")
    print(f"Chunking {len(text)} characters")
    
    chunks = []
    start = 0

    while start < len(words):
        end = start + size

        current = words[start:end]

        chunk = " ".join(current)

        chunks.append(chunk)

        start += size - overlap
    
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")
    
def semantic_chunk_command(text: str, size: int=MAX_SIZE_SEMANTIC_CHUNK, overlap: int=DEFAULT_OVERLAP_SIZE):
    cleaned_text = text.strip()

    if not cleaned_text:
        return []
    
    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
    
    if len(sentences) == 1 and not cleaned_text.endswith(('?', '!', '.')):
        sentences = [cleaned_text]

    chunks = []
    start = 0

    while start < len(sentences):
        end = start + size
        current = sentences[start:end]
        if chunks and len(stripped_list) <= overlap:
            break

        if not current:
            break

        stripped_list = []
        for s in current:
            s = s.strip()
            if s:
                stripped_list.append(s)
        if not stripped_list:
            start -= size - overlap
            continue
        chunk = " ".join(stripped_list)
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def embed_chunks_command():
    ss = ChunkedSemanticSearch()
    documents = load_movies()
    chunk_embeddings = ss.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(chunk_embeddings)} chunked embeddings")

def search_chunked_command(query: str, limit: int=SEARCH_LIMIT) -> None:
    ss = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_chunk_embeddings(documents)

    search_results = ss.search_chunks(query, limit)

    for i, result in enumerate(search_results):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"{result['document'][:100]}...")
        print("\n")