import sys
from common import model, collection

def search_documents(query, top_k=5):
    # Check whether the database contains any chunks
    total = collection.count()

    if total == 0:
        print("No documents indexed. Run ingestion first.")
        return

    # Convert the search query into an embedding
    query_embedding = model.encode(query).tolist()

    # Retrieve the most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"]
    )

    # Display the results
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        print("No results found.")
        return

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        print(f"\nResult {i}")
        print(f"File: {metadata['filename']}")
        print(f"Page: {metadata['page']}")
        print(f"Distance: {distance:.4f}")
        snippet = " ".join(document.split())
        print(f"Text: {snippet[:700]}") 
        print("-" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python search.py "your question"')
    else:
        query = " ".join(sys.argv[1:])
        search_documents(query)
