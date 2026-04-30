"""
Retriever module — finds the most relevant document chunks for a query.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document


def get_relevant_chunks(
    vector_store: Chroma,
    query: str,
    score_threshold: float = 0.3,
) -> list[Document]:
    """
    Return all chunks whose similarity score is above score_threshold.
    LangChain decides how many — no fixed k.
    Score is between 0.0 (unrelated) and 1.0 (identical).
    """
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": score_threshold},
    )
    return retriever.invoke(query)
