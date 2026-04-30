"""
Retriever module — hybrid search combining BM25 keyword matching + semantic similarity.

How it works:
  1. BM25   — classic keyword search (finds exact word matches)
  2. Semantic — ChromaDB cosine similarity (finds meaning matches)
  3. RRF    — Reciprocal Rank Fusion merges both ranked lists into one

This way "Бургер Бристол" (exact name) and "beef burger" (meaning) both work well.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def get_relevant_chunks(
    vector_store: Chroma,
    query: str,
    score_threshold: float = 0.3,
) -> list[Document]:
    """
    Hybrid search: BM25 keyword + semantic similarity, merged with RRF.
    Only returns chunks that scored in at least one of the two methods.
    """

    # ── 1. Load all stored documents (needed to build the BM25 index) ─────────
    raw = vector_store._collection.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    # ── 2. BM25 keyword search ─────────────────────────────────────────────────
    tokenized_corpus = [doc.page_content.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())

    # Rank indices highest → lowest BM25 score
    bm25_ranking = sorted(range(len(all_docs)), key=lambda i: bm25_scores[i], reverse=True)

    # ── 3. Semantic search ─────────────────────────────────────────────────────
    # Use similarity_search directly — the score_threshold parameter is used as
    # a soft guide for how many results to take (top docs by cosine similarity).
    # We fetch all docs ranked by similarity and take those above the threshold
    # relative to the best score (normalised within the result set).
    semantic_results = vector_store.similarity_search_with_relevance_scores(query, k=len(all_docs))
    # Normalise: shift scores so the best = 1.0, then filter by threshold
    if semantic_results:
        best = max(score for _, score in semantic_results)
        worst = min(score for _, score in semantic_results)
        score_range = best - worst if best != worst else 1.0
        semantic_docs = [
            doc for doc, score in semantic_results
            if (score - worst) / score_range >= score_threshold
        ]
    else:
        semantic_docs = []

    # ── 4. Reciprocal Rank Fusion (RRF) ────────────────────────────────────────
    # Formula: score += 1 / (RRF_K + rank)  for each result list
    # Higher rank (closer to 1st) = bigger contribution
    RRF_K = 60
    rrf_scores: dict[str, float] = {}

    for rank, idx in enumerate(bm25_ranking):
        content = all_docs[idx].page_content
        rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, doc in enumerate(semantic_docs):
        content = doc.page_content
        rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (RRF_K + rank + 1)

    # ── 5. Build final list ────────────────────────────────────────────────────
    # Candidates = semantic hits  +  any doc with a BM25 score > 0
    doc_by_content = {doc.page_content: doc for doc in all_docs}
    candidates: set[str] = {doc.page_content for doc in semantic_docs}
    for idx in bm25_ranking:
        if bm25_scores[idx] > 0:
            candidates.add(all_docs[idx].page_content)

    # Sort candidates by their combined RRF score
    ranked = sorted(candidates, key=lambda c: rrf_scores.get(c, 0.0), reverse=True)

    return [doc_by_content[content] for content in ranked]
