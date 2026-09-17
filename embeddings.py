""" Generate vector embeddings for chunks using a shared HuggingFace model. """

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embedding_model

def embed_chunks(chunks):
    model = get_embedding_model()
    texts = [chunk.page_content for chunk in chunks]
    vectors = model.embed_documents(texts)
    return vectors

if __name__ == "__main__":
    from ingestion import load_all_pdfs_cached
    from config import DATA_DIR
    from chunking import get_chunks

    docs = load_all_pdfs_cached(DATA_DIR)
    chunks = get_chunks(docs)
    vectors = embed_chunks(chunks[:5])
    print(f"Generated {len(vectors)} vectors")
    print(f"Vector dimension: {len(vectors[0])}")

