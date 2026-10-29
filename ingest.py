import sys
from pathlib import Path
import pymupdf
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "Data")
INDEX_DIR = "index"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=INDEX_DIR)
collection = client.get_or_create_collection("studyvault")

documents = []
metadatas = []
ids = []

for file in DATA_DIR.rglob("*.pdf"):
    try:
        pdf = pymupdf.open(file)

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if not text:
                continue

            # Split long pages into manageable chunks
            for chunk_num, start in enumerate(range(0, len(text), 800)):
                chunk = text[start:start + 800]

                documents.append(chunk)
                metadatas.append({
                    "filename": file.name,
                    "page": page_num
                })
                ids.append(
                    f"{file.name}_{page_num}_{chunk_num}"
                )

        pdf.close()
        print(f"Processed: {file.name}")

    except Exception as e:
        print(f"Failed: {file.name} - {e}")

if documents:
    embeddings = model.encode(
        documents,
        batch_size=64,
        show_progress_bar=True
    ).tolist()

    # Upsert allows repeated runs without duplicate IDs
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"\nIndexed {len(documents)} text chunks.")
else:
    print("No readable PDF text found.")