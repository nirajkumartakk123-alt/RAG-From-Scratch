import numpy as np
from sentence_transformers import SentenceTransformer
from chunking import load_documents, build_chunks

MODEL_NAME = "all-MiniLM-L6-v2"

def embed_texts(model, texts):
    return model.encode(texts, normalize_embeddings = True)

def cos_sim(p, q):
    return np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))

if __name__ == "__main__":
    chunks = build_chunks(load_documents("data"))
    model = SentenceTransformer(MODEL_NAME)

    vectors = embed_texts(model, [c["text"] for c in chunks])
    print("shape", vectors.shape, "dtype", vectors.dtype)
    print("norm of vector 0:", np.linalg.norm(vectors[0]))

    queries = [
        "Which models are examples of LLMs?",
        "How do i bake sourdough bread?",
    ]

    q_vecs = embed_texts(model, queries)
    for q, qv in zip(queries, q_vecs):
        print("\nQuery:", q)
        for c, v in zip(chunks, vectors):
            print(" ", c["id"], round(float(cos_sim(qv, v)), 3))