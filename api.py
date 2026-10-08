"""FastAPI wrapper exposing the RAG pipeline as a web service."""

from fastapi import FastAPI
from pydantic import BaseModel

from hybrid_retrieval import hybrid_retriever, retrieve_hybrid
from reranking import rerank
from generation import build_prompt, call_llm
from ingestion import load_all_pdfs_cached
from chunking import get_chunks
from config import DATA_DIR

app = FastAPI()

documents = load_all_pdfs_cached(DATA_DIR)
chunks = get_chunks(documents)
retriever = hybrid_retriever(chunks)


class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(request: Question):
    question = request.question
    retrieved = retrieve_hybrid(retriever, question)
    retrieved = rerank(question, retrieved, top_n=3)
    prompt = build_prompt(retrieved, question)
    answer = call_llm(prompt)
    sources = [chunk.metadata["source"] for chunk in retrieved]

    return {"answer": answer, "sources": sources}
