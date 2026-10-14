from pathlib import Path

CHUNK_SIZE = 500
OVERLAP = 100

def load_documents(folder):
    docs = []
    for path in sorted(Path(folder).glob("*.txt")):
        text = path.read_text(encoding = "utf-8")
        docs.append({"source": path.name, "text": text})
    return docs

def chunk_text(text, chunk_size= CHUNK_SIZE, overlap = OVERLAP):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end>= len(text):
            break
        start += step
    return chunks

def build_chunks(docs):
    all_chunks = []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc["text"])):
            all_chunks.append({
                "id": f'{doc["source"]}::{i}',
                "source": doc["source"],
                "text": piece,
            })
    return all_chunks

if __name__ == '__main__':
    docs = load_documents("data")
    chunks = build_chunks(docs)
    print(len(docs), "docs ->", len(chunks), "chunks")
    for c in chunks[:3]:
        print(c["id"], len(c["text"]), repr(c["text"][:60]))