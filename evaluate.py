from rank_bm25 import BM25Okapi
from common import model, collection
import re

# Test queries and the filenames expected to be relevant
TEST_QUERIES = [
    ("round robin scheduling", "Operating-System-.pdf"),
    ("database normalization", "Dbmsnotes.pdf"),
    ("binary search algorithm", "Cormen Introduction to Algorithms.pdf"),
    ("functional dependencies", "Dbmsnotes.pdf"),
    ("operating system process scheduling", "Operating-System-.pdf"),
]

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Load all indexed chunks for the keyword-search baseline
data = collection.get(include=["documents", "metadatas"])
documents = data["documents"]
metadatas = data["metadatas"]

if not documents:
    print("No indexed documents found. Run ingestion first.")
    raise SystemExit

bm25 = BM25Okapi([tokenize(doc) for doc in documents])

def bm25_search(query, k=3):
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [metadatas[i]["filename"] for i in ranked[:k]]

def semantic_search(query, k=3):
    vector = model.encode(query).tolist()
    result = collection.query(
        query_embeddings=[vector],
        n_results=k,
        include=["metadatas"]
    )
    return [m["filename"] for m in result["metadatas"][0]]

def evaluate():
    bm25_top1 = 0
    bm25_top3 = 0
    semantic_top1 = 0
    semantic_top3 = 0

    for query, expected_file in TEST_QUERIES:
        keyword_results = bm25_search(query)
        semantic_results = semantic_search(query)

        bm25_top1 += bool(keyword_results and keyword_results[0] == expected_file)
        bm25_top3 += expected_file in keyword_results
        semantic_top1 += bool(semantic_results and semantic_results[0] == expected_file)
        semantic_top3 += expected_file in semantic_results

        print(f"\nQuery: {query}")
        print(f"Expected file: {expected_file}")
        print(f"BM25 top 3: {keyword_results}")
        print(f"Semantic top 3: {semantic_results}")

    total = len(TEST_QUERIES)
    print("\n--- Evaluation Results ---")
    print(f"BM25 Top-1 accuracy: {bm25_top1 / total:.1%}")
    print(f"BM25 Top-3 accuracy: {bm25_top3 / total:.1%}")
    print(f"Semantic Top-1 accuracy: {semantic_top1 / total:.1%}")
    print(f"Semantic Top-3 accuracy: {semantic_top3 / total:.1%}")

if __name__ == "__main__":
    evaluate()