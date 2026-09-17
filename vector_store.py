"""Building and loading the Chroma vector store."""

from langchain_chroma import Chroma
from config import CHROMA_PERSIST_DIR
from embeddings import get_embedding_model

def build_vectorstore(chunks):
    embedding_model = get_embedding_model()
    vectorstore = Chroma.from_documents(
        embedding = embedding_model,
        documents = chunks,
        persist_directory = CHROMA_PERSIST_DIR,
    )
    return vectorstore

def get_vectorstore():
    return Chroma(
        embedding_function = get_embedding_model(),
        persist_directory = CHROMA_PERSIST_DIR,
    )

if __name__ == "__main__":
    from ingestion import load_all_pdfs_cached
    from config import DATA_DIR
    from chunking import get_chunks

    docs = load_all_pdfs_cached(DATA_DIR)
    chunks = get_chunks(docs)
    vectorstore = build_vectorstore(chunks[:5])
    print("Vectorstore built.")

    store = get_vectorstore()
    results = store.similarity_search("attention mechanism", k=2)
    for r in results:
        print(r.metadata["source"], "-", r.page_content[:100])
