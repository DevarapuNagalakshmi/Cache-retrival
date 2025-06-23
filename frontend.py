import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# 1️ Load Embedding Model (only once)
@st.cache_resource  # Streamlit will load this only once
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()
embedding_dimension = 384

# 2️ Initialize FAISS vector store (in memory for now)
if "index" not in st.session_state:
    st.session_state.index = faiss.IndexFlatL2(embedding_dimension)
    st.session_state.qa_store = []

# 3️ Create LLM (Groq)
@st.cache_resource
def load_llm():
    return ChatGroq(model="deepseek-r1-distill-llama-70b")

llm = load_llm()

# 4️ Prompt template
prompt_text = """YOU ARE A ENGINEER AT OPEN AI.
Below the question:
{Question}"""
prompt = PromptTemplate(
    input_variables=["Question"],
    template=prompt_text
)
chain = prompt | llm

# 5️ Semantic Search Function
def semantic_search(user_question, threshold=0.75):
    user_emb = embedding_model.encode(user_question).reshape(1, -1)
    if len(st.session_state.qa_store) == 0:
        return None, user_emb

    D, I = st.session_state.index.search(user_emb, k=1)
    similarity = 1 - D[0][0]

    if similarity >= threshold:
        cached_answer = st.session_state.qa_store[I[0][0]]['answer']
        return cached_answer, user_emb
    else:
        return None, user_emb

#  Streamlit UI
st.set_page_config(page_title="Cache Augmented Chatbot", layout="centered")
st.title(" Cache-Augmented Chatbot")
st.markdown("Ask your question. I'll reuse previous answers if I have seen similar ones!")

user_input = st.text_input("Enter your question:")

if st.button("Get Answer") and user_input.strip():
    with st.spinner("Thinking..."):
        cached_answer, user_emb = semantic_search(user_input)

        if cached_answer:
            st.success(" [From Cache]")
            st.write(cached_answer)
        else:
            result = chain.invoke({"Question": user_input})
            answer = result.content
            st.success("[From LLM]")
            st.write(answer)

            # Save to cache
            st.session_state.index.add(user_emb)
            st.session_state.qa_store.append({
                'question': user_input,
                'answer': answer,
                'embedding': user_emb.flatten()
            })

st.caption(" powered by AI team ")
