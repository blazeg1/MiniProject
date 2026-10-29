from sentence_transformers import SentenceTransformer
import chromadb

# Load the same embedding model used during ingestion
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

# Connect to the existing persistent database
client = chromadb.PersistentClient(path="index")

# Access the existing collection
collection = client.get_or_create_collection(
    name="studyvault"
)
