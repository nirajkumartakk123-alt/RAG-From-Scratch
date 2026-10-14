import numpy as np
from sentence_transformers import SentenceTransformer
from chunking import load_documents, build_chunks
from embedding import MODEL_NAME, embed_texts

def build_index(folder = "data"):
    chunks = build_chunks(load_documents(folder))
    model = SentenceTransformer(MODEL_NAME)
    vectors = embed_texts(model, [c["text"] for c in chunks])
    return model, chunks, vectors

def retrieve(query, model, chunks, vectors, k = 3):
    q_vec = embed_texts(model, [query])[0]
    scores = vectors @ q_vec
    top_idx = np.argsort(scores)[::-1][:k]
    return [
        {"id": chunks[i]["id"], "score": float(scores[i]), "text":chunks[i]["text"]}
        for i in top_idx
    ]

if __name__ == "__main__":
    model, chunks, vectors = build_index()
    print(len(chunks), "chunks, vectors shape:", vectors.shape)

    queries = [
    # --- ANSWERABLE (expected source file in comment) ---
    "What are queries, keys, and values in self-attention?",            # transformers
    "What is BM25 and how does sparse retrieval differ from dense?",    # rag
    "What is an agent loop and who controls it?",                       # agents
    "How is cosine similarity calculated?",                             # embeddings
    "What does a reranker do?",                                         # rag (primary), embeddings mentions it briefly

    # --- UNANSWERABLE ---
    "Who won the 2022 FIFA World Cup?",                                 # off-topic
    "How does LoRA fine-tuning work?",                                  # near-miss: fine-tuning is mentioned, LoRA is not
    "What is the per-token price of the GPT-4 API?",                    # AI-adjacent, not in any file
    "as in the of and to",
    "Transformers attention keys values"
]
    for q in queries:
        print("\nQuery:", q)
        for r in retrieve(q, model, chunks, vectors, k = 3):
            print(f' {r["score"]:.3f}  {r["id"]}  {r["text"][:70]!r}')