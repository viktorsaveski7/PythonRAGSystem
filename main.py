"""
RAG System - FastAPI entry point

A simple Retrieval-Augmented Generation pipeline exposed as a REST API:
  POST /ingest   — load docs from the docs/ folder and build the vector store
  POST /query    — ask a question and get back retrieved context (+ LLM answer if wired up)
  GET  /health   — liveness check

Run with:
  uvicorn main:app --reload
"""

import os
import shutil

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.loader import load_documents, load_from_urls
from rag.vector_store import build_vector_store, load_vector_store
from rag.retriever import get_relevant_chunks
from rag.generator import generate_answer


DOCS_DIR = "docs"
DB_DIR = "chroma_db"

app = FastAPI(title="RAG System API", version="1.0.0")


# ── Request / Response models ────────────────────────────────────────────────

class IngestUrlRequest(BaseModel):
    urls: list[str]


class QueryRequest(BaseModel):
    question: str
    score_threshold: float = 0.3


class QueryResponse(BaseModel):
    question: str
    chunks_retrieved: int
    score_threshold: float
    search_method: str
    answer: str


class IngestResponse(BaseModel):
    message: str
    documents_ingested: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.delete("/reset")
def reset():
    """Delete all vectors from the store so you can ingest fresh data."""
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    return {"message": "Vector store cleared."}


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Load documents from the docs/ folder and build the vector store."""
    try:
        documents = load_documents(DOCS_DIR)
        build_vector_store(documents, persist_directory=DB_DIR)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return IngestResponse(
        message="Vector store built and saved.",
        documents_ingested=len(documents),
    )


@app.post("/ingest-url", response_model=IngestResponse)
def ingest_url(request: IngestUrlRequest):
    """Scrape one or more web pages and add them to the vector store."""
    try:
        documents = load_from_urls(request.urls)
        build_vector_store(documents, persist_directory=DB_DIR)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        message=f"Scraped and ingested {len(request.urls)} URL(s).",
        documents_ingested=len(documents),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Query the RAG system with a question."""
    try:
        vector_store = load_vector_store(persist_directory=DB_DIR)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Vector store not found. Run POST /ingest first.",
        )

    chunks = get_relevant_chunks(vector_store, request.question, score_threshold=request.score_threshold)
    answer = generate_answer(request.question, chunks)

    return QueryResponse(
        question=request.question,
        chunks_retrieved=len(chunks),
        score_threshold=request.score_threshold,
        search_method="hybrid (BM25 + semantic, merged with RRF)",
        answer=answer,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
