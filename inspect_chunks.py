import chromadb

client = chromadb.PersistentClient(path="chroma_db")
col = client.list_collections()[0]
print(f"Total chunks: {col.count()}\n")

result = col.get(include=["documents", "metadatas", "embeddings"])

for i, (doc, meta, emb) in enumerate(zip(result["documents"], result["metadatas"], result["embeddings"])):
    source = meta.get("source", "?")
    print(f"--- CHUNK {i+1} [{source}] ---")
    print(doc[:400])
    print(f"Vector dims: {len(emb)}  |  first 8 values: {[round(x, 4) for x in emb[:8]]}")
    print()
