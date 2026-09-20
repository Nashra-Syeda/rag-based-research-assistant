""" Test all the eval questions """

from eval_questions import EVAL_QUESTIONS
from vector_store import get_vectorstore
from generation import build_prompt, call_llm

vectorstore = get_vectorstore()

for item in EVAL_QUESTIONS:

    print("\n" + "=" * 80)

    question = item["question"]
    chunks = vectorstore.similarity_search(question, k =3)
    actual_sources = [chunk.metadata["source"] for chunk in chunks]
    expected_source = item["expected_source"]
    prompt = build_prompt(chunks,question)
    answer = call_llm(prompt)
    
    print("Question:", question)
    print("Expected_source:", expected_source)
    print("Actual_source:", actual_sources)
    print("Generated_answer:", answer)







