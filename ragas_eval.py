"""ragas_eval.py — automated RAG evaluation using Ragas (faithfulness, answer relevancy, context precision/recall)."""

import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq

from eval_questions import EVAL_QUESTIONS
from hybrid_retrieval import hybrid_retriever, retrieve_hybrid
from reranking import rerank
from generation import build_prompt, call_llm
from ingestion import load_all_pdfs_cached
from chunking import get_chunks
from embeddings import get_embedding_model
from config import DATA_DIR, GROQ_MODEL_NAME

load_dotenv()

judge_llm = LangchainLLMWrapper(ChatGroq(model=GROQ_MODEL_NAME, api_key=os.getenv("GROQ_API_KEY"), max_tokens=8192))
judge_embeddings = LangchainEmbeddingsWrapper(get_embedding_model())

documents = load_all_pdfs_cached(DATA_DIR)
chunks = get_chunks(documents)
retriever = hybrid_retriever(chunks)

data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

for item in EVAL_QUESTIONS:
    question = item["question"]
    retrieved = retrieve_hybrid(retriever, question)
    retrieved = rerank(question, retrieved, top_n=3)
    prompt = build_prompt(retrieved, question)
    answer = call_llm(prompt)

    data["question"].append(question)
    data["answer"].append(answer)
    data["contexts"].append([chunk.page_content for chunk in retrieved])
    data["ground_truth"].append(item["reference_answer"])

dataset = Dataset.from_dict(data)

metrics = [
    Faithfulness(llm=judge_llm),
    AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1),
    ContextPrecision(llm=judge_llm),
    ContextRecall(llm=judge_llm),
]

run_config = RunConfig(max_workers=2, timeout=180)

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    run_config=run_config,
)

print(result)