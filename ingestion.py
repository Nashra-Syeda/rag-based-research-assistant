""" Load PDFs from the data folder into LangChain Documents. """ 

from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

from config import DATA_DIR

MIN_CHARS_THRESHOLD = 1000

_converter = None 

def _get_converter(): 
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter

def load_pdf(pdf_path):
    converter = _get_converter()
    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()
    return Document(page_content=markdown, metadata={"source": pdf_path})
    
def load_all_pdfs(data_dir):
    pdf_paths = list(Path(data_dir).glob("*.pdf"))
    documents = [ ]
    for path in pdf_paths:
        print(f"Processing {path.name}...")
        doc = load_pdf(path)
        if len(doc.page_content) < 1000:
            print(f"warning: {path} extracted only {len(doc.page_content)} characters - likely a parse failure")
        documents.append(doc)
    return documents 

if __name__ == "__main__":
    docs = load_all_pdfs(DATA_DIR)
    print(f"Loaded {len(docs)} documents")
    for d in docs:
        print(d.metadata["source"], "-", len(d.page_content), "chars")