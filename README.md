# rag-from-scratch

A question-answering assistant over a small set of text files, built step by step
without LangChain to understand each stage of a RAG pipeline: chunking, local
embeddings (all-MiniLM-L6-v2), cosine-similarity retrieval, grounded prompt
construction, and answer generation with a local LLM (llama3.2:3b via Ollama).

Status: core pipeline working. Planned: FastAPI serving, a retrieval and
generation evaluation harness (Recall@k, false-refusal and hallucination
rates), and upgrades such as sentence-aware chunking, similarity thresholds,
reranking and tool calling.