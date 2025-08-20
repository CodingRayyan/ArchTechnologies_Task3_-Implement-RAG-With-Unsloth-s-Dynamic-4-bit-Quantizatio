import os, sys
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Fix sqlite3 issue for ChromaDB on Streamlit Cloud
# -------------------------------
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import streamlit as st

# -------------------------------
# Dependency loading
# -------------------------------
try:
    from transformers import pipeline
    import torch
    import chromadb
    DEPENDENCIES_LOADED = True
except Exception as e:
    st.error(f"Failed to load dependencies: {e}")
    DEPENDENCIES_LOADED = False

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    layout="centered"
)

st.title("🤖 RAG Chatbot with Instruction-Tuned Model - Developed by Rayyan Ahmed")
st.write("Ask me anything, I’ll search your knowledge base + use an instruction-tuned model for answers.")

# -------------------------------
# Initialize ChromaDB (only if deps are loaded)
# -------------------------------
if DEPENDENCIES_LOADED:
    try:
        client = chromadb.Client()
        st.success("ChromaDB client initialized successfully ✅")
    except Exception as e:
        st.error(f"ChromaDB initialization failed: {e}")

# -------------------------------
# Chat UI
# -------------------------------
user_question = st.text_input("Enter your question:")

if user_question:
    if not DEPENDENCIES_LOADED:
        st.error("Dependencies are missing. Please fix installation.")
    else:
        try:
            # Example pipeline (replace with your model later)
            qa_pipeline = pipeline("text-generation", model="gpt2")
            answer = qa_pipeline(user_question, max_length=100, num_return_sequences=1)[0]['generated_text']
            st.write("### Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating answer: {e}")

