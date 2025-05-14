from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> list[float]:
    return embedder.encode(text, normalize_embeddings=True).tolist()


client = chromadb.PersistentClient(path="./embeddings")

collection = client.get_or_create_collection(
    name="dialogue_memory",
    metadata={"description": "agent turns embeddings"}
)


def store_turn(turn_id: str, text: str, metadata: dict):
    emb = get_embedding(text)
    collection.add(
        ids=[turn_id],
        embeddings=[emb],
        documents=[text],
        metadatas=[metadata]
    )

def retrieve_similar(text: str, n_results: int = 5):
    emb = get_embedding(text)
    results = collection.query(
        query_embeddings=[emb],
        n_results=n_results
    )
    return results['documents'][0], results['metadatas'][0]
