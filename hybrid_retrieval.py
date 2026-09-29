""" Retrieve the chunks using Hybrid retrieval(semantic search + keyword search). """

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from vector_store import get_vectorstore

def bm25_retriever(chunks):
    retriever = BM25Retriever.from_documents(chunks)
    return retriever

def retrieve_hybrid(retriever, question):
    results = retriever.invoke(question)
    return results

def hybrid_retriever(chunks):
    bm25 = bm25_retriever(chunks)
    vectorstore = get_vectorstore()
    semantic = vectorstore.as_retriever()
    Ensemble = EnsembleRetriever(
        retrievers = [bm25, semantic],
        weights = [0.5, 0.5]
    )
    return Ensemble


if __name__ == "__main__":
    from ingestion import load_all_pdfs_cached
    from config import DATA_DIR
    from chunking import get_chunks
    documents = load_all_pdfs_cached(DATA_DIR)
    chunks = get_chunks(documents)
    retriever = hybrid_retriever(chunks)
    question = "What was the BLEU score reported for the English-to-German translation task?"
    results = retrieve_hybrid(retriever, question)
    print(results)


