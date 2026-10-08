# RAG Research Paper Assistant

A retrieval-augmented question-answering system over a small corpus of foundational NLP/LLM papers. Ask a natural-language question and get a grounded answer with the source paper cited. The pipeline combines keyword and semantic retrieval, reranks the candidates with a cross-encoder, and generates the final answer with an LLM that is instructed to stay inside the retrieved context.

It is exposed as a FastAPI service and evaluated two ways: a hand-verified question set and automated Ragas metrics.

## Architecture

```
PDFs --> Docling (PDF to Markdown) --> Recursive chunking (Markdown-aware)
                                              |
                      +-----------------------+-----------------------+
                      v                                               v
            BM25 keyword index                         Embeddings (all-MiniLM-L6-v2)
                                                              |
                                                       Chroma vector store
                      +-----------------------+-----------------------+
                                              v
                              Hybrid retrieval (BM25 + dense, fused with RRF)
                                              v
                       Cross-encoder reranking (ms-marco-MiniLM-L-6-v2, top 3)
                                              v
                       Prompt with sourced context --> Groq LLM --> answer + sources
```

| Stage | Choice | Notes |
|---|---|---|
| Parsing | Docling | Structure-aware, outputs Markdown so headings survive |
| Chunking | `RecursiveCharacterTextSplitter`, 700 chars, 100 overlap | Separators put Markdown headings (`## `, `### `) ahead of paragraph breaks |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, loaded once and shared |
| Vector store | Chroma (persisted to disk) | Build and load are separate steps |
| Keyword search | BM25 (LangChain `BM25Retriever`) | Catches exact terms and names that embeddings can blur |
| Fusion | LangChain `EnsembleRetriever`, weights 0.5 / 0.5 | Weighted reciprocal rank fusion |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Scores each (query, chunk) pair jointly; keeps top 3 |
| Generation | Groq `openai/gpt-oss-20b` | Prompt forbids outside knowledge and requires source mentions |
| Serving | FastAPI + uvicorn | `POST /ask` |

The corpus is 10 papers, which becomes 2,975 chunks after splitting.

## Corpus

Attention Is All You Need, BERT, RoBERTa, ELECTRA, GPT-3, InstructGPT, T5, LoRA, Sentence-BERT, and the original RAG paper (Lewis et al.). They overlap enough (BERT / RoBERTa / ELECTRA / Sentence-BERT in particular) that retrieval has to discriminate between similar papers, not just find the one obvious match.

The PDFs are not committed. Download them from arXiv into a `data/` folder:

| Paper | arXiv |
|---|---|
| Attention Is All You Need | 1706.03762 |
| BERT | 1810.04805 |
| GPT-3 | 2005.14165 |
| RoBERTa | 1907.11692 |
| T5 | 1910.10683 |
| LoRA | 2106.09685 |
| RAG (Lewis et al.) | 2005.11401 |
| InstructGPT | 2203.02155 |
| ELECTRA | 2003.10555 |
| Sentence-BERT | 1908.10084 |

Save them as `data/<name>.pdf` (for example `data/BERT.pdf`, `data/attention is all you need.pdf`). The evaluation set refers to these filenames.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

The first run parses every PDF with Docling, which is slow. The parsed documents are cached to `docs_cache.pkl`, so later runs load instantly. Delete that file if you change the PDFs or the ingestion code.

Build the vector store once (all chunks, not a test slice):

```bash
python vector_store.py
```

## Run the API

```bash
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive UI, or call it directly:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the rank of the low-rank matrices used in LoRA?"}'
```

Response:

```json
{
  "answer": "...",
  "sources": ["data\\LoRA.pdf", "data\\LoRA.pdf", "data\\LoRA.pdf"]
}
```

The retriever is built once at server start, not per request.

## Project layout

| File | Role |
|---|---|
| `config.py` | Chunk size, model names, retrieval settings, paths |
| `ingestion.py` | Docling PDF loading, length validation, pickle cache |
| `chunking.py` | Markdown-aware recursive splitting, metadata preserved |
| `embeddings.py` | Shared HuggingFace embedding model |
| `vector_store.py` | Build and load the Chroma store |
| `hybrid_retrieval.py` | BM25 + dense retrieval fused with `EnsembleRetriever` |
| `reranking.py` | Cross-encoder reranker |
| `generation.py` | Prompt construction and Groq call |
| `api.py` | FastAPI service |
| `eval_questions.py` | Eval set: question, expected source, reference answer |
| `run_eval.py` | Runs the eval set through the full pipeline and prints results |
| `ragas_eval.py` | Automated Ragas scoring |

## Evaluation

### Manual evaluation (8 questions)

The eval set was written by hand, and the expected sources were checked against the papers themselves. It includes single-paper factual questions, questions that need the right paper out of several similar ones (ELECTRA vs BERT, RoBERTa vs BERT), a multi-hop question, and two questions with no answer in the corpus (GPT-4 in the InstructGPT paper, ImageNet state of the art), where the correct behavior is to refuse.

I ran the same 8 questions against three pipeline stages:

| Pipeline | Result |
|---|---|
| Dense retrieval only | 7/8 on first run, then 8/8 after fixing the prompt |
| + BM25 hybrid retrieval | 8/8 |
| + cross-encoder reranking | 8/8 |

Pass rate did not change between the three retrieval configurations, because this small question set was already answerable with dense retrieval alone. What did change: hybrid retrieval returned more supporting chunks per query, and reranking narrowed them back to the most relevant three (visible in a before/after check, where it dropped an off-topic T5 chunk and reordered the rest).

**A real failure caught along the way.** On the ImageNet question, the first pipeline retrieved unrelated T5 chunks and then answered with a confident, detailed result about ResNet-152, including numbers and a citation that were not in any retrieved context. The prompt said to use the context but never said what to do when the context was irrelevant. Adding an explicit instruction (answer only from the context, otherwise reply with a fixed refusal sentence) fixed it, and both out-of-corpus questions now refuse correctly.

An earlier bug is also worth noting: for a while every query returned chunks from one paper, because the vector store had only ever been built from a 5-chunk test slice. The fix was building from the full chunk list.

### Automated evaluation (Ragas)

Judged with the same Groq model used for generation, over the same 8 questions, with a reference answer written for each.

| Metric | Score | Notes |
|---|---|---|
| Context precision | 0.7812 | 1 of 16 judge calls timed out |
| Context recall | 0.7143 | 1 of 16 judge calls timed out |
| Faithfulness | 0.4375 -> 0.4652 | After tightening the grounding instruction; 1 of 8 judge calls failed on the second run |
| Answer relevancy | 0.5642 | Measured before the prompt change, not re-measured |

Reading these together: retrieval is doing its job (relevant chunks are found and ranked reasonably), while generation grounding is the weak spot. Answers that were correct on manual review were still penalized by faithfulness when the model added background it knew but the retrieved chunks did not state, for example explaining BERT's objective by contrast when asked about ELECTRA. Tightening the prompt to forbid unsupported background moved faithfulness only slightly (0.4375 to 0.4652), which suggests the model's tendency to add background is not fixed by wording alone.

## Limitations

- The eval set is 8 questions. The manual pass rate hit its ceiling quickly and says little about harder queries; the Ragas numbers are the more informative signal, and they are noisy at this sample size.
- Ragas used the same model as the generator for judging, on a free-tier Groq account. Several judge calls failed or timed out, and I report the scores with those failures noted rather than treating them as clean. Judge-model choice affects the numbers.
- Answer relevancy was not re-run after the prompt change, so it is not directly comparable to the faithfulness figure.
- Faithfulness is moderate. Fixing it properly would likely need a stronger instruction-following model or a post-generation grounding check.
- Chunk size (700), embedding model, and fusion weights (0.5 / 0.5) were set from a prior project and not tuned against this eval set.
- Source citations are file paths from the chunk metadata.

## Possible next steps

- Containerize with Docker.
- Tune chunk size and fusion weights against a larger eval set.
- Add a post-generation grounding check to improve faithfulness.
- Fetch papers live from the arXiv API instead of using a fixed corpus.