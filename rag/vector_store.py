"""
Vector store module using ChromaDB with local sentence-transformer embeddings.

No API key is required — embeddings run locally on your machine.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# A small, fast model that runs fully offline (~90 MB download on first use)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(
    documents: list[Document],
    persist_directory: str = "chroma_db",
) -> Chroma:
    """Embed documents and persist them in a local ChromaDB collection."""
    embeddings = _get_embeddings()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    return vector_store


def load_vector_store(persist_directory: str = "chroma_db") -> Chroma:
    """Load an existing ChromaDB collection from disk."""
    embeddings = _get_embeddings()

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
