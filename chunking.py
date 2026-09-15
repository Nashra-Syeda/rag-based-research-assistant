""" Split the Documnets into smaller chunks for embedding. """

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def get_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n##", "\n###", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    from ingestion import load_all_pdfs_cached
    from config import DATA_DIR

    docs = load_all_pdfs_cached(DATA_DIR)
    chunks = get_chunks(docs)
    print(f"Loaded {len(docs)} documents ->  {len(chunks)} chunks")
    print(chunks[0].page_content)
    print(chunks[0].metadata)



