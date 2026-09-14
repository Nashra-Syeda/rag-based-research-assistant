DATA_DIR = "data"

# Chunking
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-20b"

# Retrieval
DEFAULT_TOP_K = 5
DEFAULT_RERANK_TOP_N = 3
RRF_K = 60