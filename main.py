from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# 1️ Load Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dimension = 384  # MiniLM output dimension

# 2️ Initialize FAISS vector store
index = faiss.IndexFlatL2(embedding_dimension)
qa_store = []  # List to store {'question': ..., 'answer': ..., 'embedding': ...}

# 3️ Create LLM with Groq
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# 4️ Prompt template
text = """YOU ARE A ENGINEER AT OPEN AI.
Below the question
{Question}"""
prompt = PromptTemplate(
    input_variables=["Question"],
    template=text
)
chain = prompt | llm

# 5️ Semantic Search Function
def semantic_search(user_question, threshold=0.75):
    user_emb = embedding_model.encode(user_question)
    if len(qa_store) == 0:
        return None, user_emb
    D, I = index.search(np.array([user_emb]), k=1)
    if D[0][0] < (1 - threshold):
        return qa_store[I[0][0]]['answer'], user_emb
    else:
        return None, user_emb

# 6️ Main Chat Loop
while True:
    user_input = input("Enter your Question (or 'exit'): ")
    if user_input.lower() == 'exit':
        break

    cached_answer, user_emb = semantic_search(user_input)

    if cached_answer:
        print(f"[From Cache] {cached_answer}")
    else:
        result = chain.invoke({"Question": user_input})
        answer = result.content

        print(f"[From LLM] {answer}")

        # Save to cache
        index.add(np.array([user_emb]))
        qa_store.append({'question': user_input, 'answer': answer, 'embedding': user_emb})
