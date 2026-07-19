import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import SEARCH_LIMIT, ALPHA

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def normalize_command(scores: list[float]) -> None:
    if not scores:
        return
    
    minimum = min(scores)
    maximum = max(scores)

    scores_list = []

    if minimum == maximum:
        scores_list = [1.0] * len(scores)
    else:
        for score in scores:
            normalized_score = (score - minimum) / (maximum - minimum)
            scores_list.append(normalized_score)

    for score in scores_list:
        print(f"* {score:.4f}")

def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_command(query: str, alpha: float=ALPHA, limit: int=SEARCH_LIMIT):
    pass

def rrf_search_command():
    pass