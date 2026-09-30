""" Cross-Encoder Reranking to narrow hybrid retrieval results to the most relevant chunks. """

from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL_NAME

_reranker = None

def _get_reranker():
    global _reranker
    if _reranker == None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker

def rerank(query, chunks, top_n):
    pairs = [[query, chunk.page_content] for chunk in chunks]
    scores = _get_reranker().predict(pairs)
    scored_chunks = list(zip(scores,chunks))
    scored_chunks.sort(key=lambda  x: x[0], reverse =True)
    return [item[1] for item in scored_chunks[:top_n]]

if __name__ == "__main__":
    from config import DATA_DIR
    from ingestion import load_all_pdfs_cached
    from chunking import get_chunks
    from hybrid_retrieval import retrieve_hybrid, hybrid_retriever

    documents = load_all_pdfs_cached(DATA_DIR)
    chunks = get_chunks(documents)
    retriever = hybrid_retriever(chunks)
    question = "What was the BLEU score reported for the English-to-German translation task"
    results = retrieve_hybrid(retriever, question)
    print("BEFORE:")
    for i, chunk in enumerate(results):
        print(i, chunk.page_content[:100].replace("\n", " "))
    reranked = rerank(question, results, 3)
    print("AFTER:")
    for i, chunk in enumerate(reranked):
        print(i, chunk.page_content[:100].replace("\n", " "))



