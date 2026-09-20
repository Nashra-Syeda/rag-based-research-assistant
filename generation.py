
""" Build a prompt from retrieved chunks and generate an answer via Groq """

import os
from groq import Groq
from dotenv import load_dotenv
from config import GROQ_MODEL_NAME

def build_prompt(chunks, question):
    context = ""

    for chunk in chunks:
        context = context + f"{chunk.metadata['source']}\n{chunk.page_content}\n\n"

    prompt = f"""Context: {context}
Question: {question}
Instructions: Answer ONLY using the provided context above. If the context does not contain information relevant to answering the question, respond exactly with: "I don't have enough relevant information in the provided documents to answer this question." Do not use outside knowledge. Always mention which source(s) you used."""

    return prompt

load_dotenv()

def call_llm(prompt):

    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key = api_key)

    response = client.chat.completions.create(
        model = GROQ_MODEL_NAME,
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


    
if __name__ == "__main__":
    from vector_store import get_vectorstore 
    vectorstore = get_vectorstore()
    question = "explain attention mechanism"
    chunks = vectorstore.similarity_search(
        question,
        k = 3
    )
    prompt = build_prompt(chunks,question)
    answer = call_llm(prompt)
    print(answer)