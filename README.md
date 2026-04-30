# PythonRAGSystem

A hands-on RAG (Retrieval-Augmented Generation) implementation for learning purposes.

## Project structure

```
PythonRAGSystem/
├── docs/               ← Put your .txt or .pdf documents here
├── rag/
│   ├── loader.py       ← Loads & splits documents into chunks
│   ├── vector_store.py ← Embeds chunks and stores in ChromaDB
│   ├── retriever.py    ← Finds relevant chunks for a query
│   └── generator.py    ← Produces the final answer (LLM optional)
├── main.py             ← CLI entry point
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Activate the virtual environment
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Copy and fill in your .env
copy .env.example .env
```

## Usage

```bash
# Step 1 — add documents to docs/ then ingest them
python main.py ingest

# Step 2 — ask a question
python main.py query "What is RAG?"
```

## Going further

- **Use a real LLM**: Uncomment Option B in `rag/generator.py` and add your OpenAI key to `.env`
- **Swap the vector store**: Try FAISS or Pinecone instead of ChromaDB
- **Try different embeddings**: `text-embedding-3-small` (OpenAI) or `nomic-embed-text` (Ollama)
