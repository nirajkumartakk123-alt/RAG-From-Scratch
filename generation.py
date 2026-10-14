import requests
from retrieval import build_index, retrieve
from prompting import build_prompt

OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3.2:3b"

def ask_llm(prompt, model=LLM_MODEL):
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]

def answer(question, embed_model, chunks, vectors, k=3):
    results = retrieve(question, embed_model, chunks, vectors, k=k)
    return ask_llm(build_prompt(question, results)), results

if __name__ == "__main__":
    embed_model, chunks, vectors = build_index()
    for q in [
        "What is an agent loop and who controls it?",
        "Who won the 2022 FIFA World Cup?",
        "How does LoRA fine-tuning work?",
    ]:
        reply, results = answer(q, embed_model, chunks, vectors)
        print("\nQ:", q)
        print("Sources:", [r["id"] for r in results])
        print("A:", reply)
