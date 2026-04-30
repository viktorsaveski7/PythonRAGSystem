import chromadb

client = chromadb.PersistentClient(path="chroma_db")
col = client.list_collections()[0]
print(f"Total chunks: {col.count()}\n")

result = col.get(include=["documents", "metadatas"])

for i, (doc, meta) in enumerate(zip(result["documents"], result["metadatas"])):
    source = meta.get("source", "?")
    print(f"--- CHUNK {i+1} [{source}] ---")
    print(doc[:400])
    print()
