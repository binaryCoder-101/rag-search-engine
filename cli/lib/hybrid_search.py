import os
from typing import TypedDict

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import SEARCH_LIMIT, ALPHA, format_search_result, SearchResult, load_movies

class WeightedSearchCommandResult(TypedDict):
    original_query: str
    query: str
    alpha: float
    results: list[SearchResult]

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
        keyword_result = self._bm25_search(query, 500 *  limit)
        semantic_result = self.semantic_search.search_chunks(query, 500 * limit)

        keyword_scores = []
        semantic_scores = []

        for result in keyword_result:
            keyword_scores.append(result["score"])

        for result in semantic_result:
            semantic_scores.append(result["score"])

        normalized_keyword_scores = normalize_command(keyword_scores)
        normalized_semantic_scores = normalize_command(semantic_scores)

        documents = {}

        for result, score in zip(keyword_result, normalized_keyword_scores):
            documents[result["id"]] = {
                "title": result["title"],
                "document": result["document"],
                "keyword_score": score,
                "semantic_score": 0,
            }

        for result, score in zip(semantic_result, normalized_semantic_scores):
            if result["id"] not in documents:
                documents[result["id"]] = {
                    "title": result["title"],
                    "document": result["document"],
                    "keyword_score": 0,
                    "semantic_score": score,
                }
            else:
                documents[result["id"]]["semantic_score"] = score

        hybrid_results: list[SearchResult] = []

        for doc_id, document in documents.items():
            keyword_scr = document["keyword_score"]
            semantic_scr = document["semantic_score"]
            hybrid_scr = hybrid_score(keyword_scr, semantic_scr)
            result = format_search_result(
                doc_id=doc_id,
                title=document["title"],
                document=document["document"],
                score=hybrid_scr,
                bm25_score=keyword_scr,
                semantic_score=semantic_scr,
            )
            hybrid_results.append(result)

        sorted_documents = sorted(hybrid_results, key=lambda x: x["score"], reverse=True)

        return sorted_documents


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

    # for score in scores_list:
    #     print(f"* {score:.4f}")

    return scores_list

def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_command(
        query: str, alpha: float=ALPHA, limit: int=SEARCH_LIMIT
) -> WeightedSearchCommandResult:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }

def rrf_search_command():
    pass