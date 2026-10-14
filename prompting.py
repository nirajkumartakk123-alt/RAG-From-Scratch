from retrieval import build_index, retrieve

SYSTEM_RULES = (
    "Answer the question using ONLY the context below. "
    "If the context does not contain the answer, say you don't know. "
    "Cite sources by their [id]."
)

def build_prompt(question, results):
    context = "\n\n".join(f'[{r["id"]}]\n{r["text"]}' for r in results)
    return f"{SYSTEM_RULES}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

if __name__ == "__main__":
    model, chunks, vectors = build_index()
    q = "What is an agent loop and who controls it?"
    print(build_prompt(q, retrieve(q, model, chunks, vectors, k=3)))