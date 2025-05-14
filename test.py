# test_memory.py
from vector_db import store_turn, retrieve_similar
import uuid

def main():
   
    samples = [
        ("Hello, how are you?", {"speaker": "AgentA", "step": 1}),
        ("I’m doing well, thanks! And you?", {"speaker": "AgentB", "step": 2}),
        ("What do you think about cloud rights?", {"speaker": "AgentA", "step": 3}),
        ("I believe clouds deserve respect but not rights.", {"speaker": "AgentB", "step": 4}),
        ("Soup memory is the idea that soup remembers its ingredients.", {"speaker": "AgentA", "step": 5}),
    ]
    for text, meta in samples:
        turn_id = str(uuid.uuid4())
        store_turn(turn_id, text, meta)
        print(f"Stored turn {turn_id[:8]}: '{text[:30]}...'")
    
    # 3. Query for similarity
    query = "How are you doing?"
    docs, metadatas = retrieve_similar(query, n_results=3)
    
    print(f"\nQuery: {query}")
    for i, (doc, meta) in enumerate(zip(docs, metadatas), start=1):
        print(f" {i}. «{doc}» — {meta}")

if __name__ == "__main__":
    main()
