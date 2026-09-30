""" Test all the eval questions """

from eval_questions import EVAL_QUESTIONS
from generation import build_prompt, call_llm
from hybrid_retrieval import hybrid_retriever, retrieve_hybrid
from ingestion import load_all_pdfs_cached
from chunking import get_chunks
from config import DATA_DIR
from reranking import rerank


documents = load_all_pdfs_cached(DATA_DIR)
chunks = get_chunks(documents)
retriever = hybrid_retriever(chunks)

for item in EVAL_QUESTIONS:

    print("\n" + "=" * 80)

    question = item["question"]
    retrieved_chunks = retrieve_hybrid(retriever, question)
    retrieved_chunks = rerank(question, retrieved_chunks, 3)
    actual_sources = [chunk.metadata["source"] for chunk in retrieved_chunks]
    expected_source = item["expected_source"]
    prompt = build_prompt(retrieved_chunks, question)
    answer = call_llm(prompt)
    
    print("Question:", question)
    print("Expected_source:", expected_source)
    print("Actual_source:", actual_sources)
    print("Generated_answer:", answer)







