import streamlit as st
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")  # MUST be first

# -------------------------------
# Background Image
# -------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)) , url("https://static.vecteezy.com/system/resources/previews/020/067/380/non_2x/chat-ai-conversation-method-background-free-vector.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import chromadb
import torch

# -------------------------------
# Sidebar Info
# -------------------------------
st.markdown("""
    <style>
    /* Sidebar custom style */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 50, 80, 0.7);  /* Dark blue-ish tone */
        color: white;
    }

    [data-testid="stSidebar"] .css-1v3fvcr {
        color: white;
    }

    /* Optional: make sidebar title/headings colored */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #00171F;  /* Light cyan */
    }

    /* Optional: control scrollbar style inside sidebar */
    ::-webkit-scrollbar-thumb {
        background: #00cfff;
        border-radius: 10px;
    }
    </style>
            
""", unsafe_allow_html=True)

with st.sidebar.expander("📁 Project Intro"):
    st.markdown(
        "- **This is a RAG Chatbot using Microsoft Phi-3.5-mini-instruct model.**\n"
        "  It searches your knowledge base and generates answers using state-of-the-art small LLM."
    )

with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("- **Hi, I'm Rayyan Ahmed**")
    st.markdown("- **IBM Certified Advanced LLM FineTuner**")
    st.markdown("- **Google Certified Soft Skills Professional**")
    st.markdown("- **Hugging Face Certified in Fundamentals of LLMs**")
    st.markdown(
        "- **Expertise:** EDA, ML, Reinforcement Learning, ANN, CNN, CV, RNN, NLP, LLMs"
    )
    st.markdown("[💼 Visit Rayyan's LinkedIn Profile](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")

with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("- **Python 3**")
    st.markdown("- **Transformers / Hugging Face** (Microsoft Phi-3.5-mini-instruct)")
    st.markdown("- **Torch / PyTorch** (for model backend, GPU support)")
    st.markdown("- **ChromaDB** (for RAG retrieval and document indexing)")
    st.markdown("- **Streamlit** (for web app interface)")
    st.markdown("- **Numpy** (numerical operations)")
    st.markdown("- **Pandas** (data handling and preprocessing)")
    st.markdown("- **RAG Architecture** (Retrieval-Augmented Generation setup)")

with st.sidebar.expander("🎯 Model Options"):
    st.markdown("**Current Model:** Microsoft Phi-3.5-mini-instruct")
    st.markdown("**Alternative Models:**")
    st.markdown("- Qwen2-1.5B-Instruct")
    st.markdown("- TinyLlama-1.1B-Chat")
    st.markdown("- Llama-3.2-1B-Instruct")
    st.markdown("- Mistral-7B-Instruct-v0.3 (if more resources)")

# -------------------------------
# Check GPU availability
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    st.sidebar.success("GPU is available ✅")
    st.sidebar.write(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("Using CPU ⚠️")

# -------------------------------
# Model Selection
# -------------------------------
MODEL_OPTIONS = {
    "Microsoft Phi-3.5-mini (Recommended)": "microsoft/Phi-3.5-mini-instruct",
    "Qwen2 1.5B": "Qwen/Qwen2-1.5B-Instruct", 
    "TinyLlama 1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Llama 3.2 1B": "meta-llama/Llama-3.2-1B-Instruct"
}

selected_model_name = st.sidebar.selectbox(
    "Choose Model:",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)
selected_model_path = MODEL_OPTIONS[selected_model_name]

# -------------------------------
# Load Model with Better Configuration
# -------------------------------
@st.cache_resource
def load_model(model_name):
    try:
        # For Phi-3.5 and similar models, use text-generation pipeline
        if "phi" in model_name.lower() or "qwen" in model_name.lower() or "llama" in model_name.lower():
            return pipeline(
                "text-generation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                return_full_text=False
            )
        else:
            # Fallback for other models
            return pipeline(
                "text-generation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                return_full_text=False
            )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Fallback to a simpler model
        return pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=-1,
            return_full_text=False
        )

# Load the selected model
with st.spinner(f"Loading {selected_model_name}..."):
    pipe = load_model(selected_model_path)

# -------------------------------
# ChromaDB Setup
# -------------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_collection")

# -------------------------------
# Enhanced RAG Query Function
# -------------------------------
def rag_query(question, max_tokens=100, temperature=0.1, top_p=0.9):
    # Retrieve relevant context
    results = collection.query(query_texts=[question], n_results=5)
    context = ""
    
    if results and results.get("metadatas") and results["metadatas"][0]:
        context_pieces = []
        for metadata in results["metadatas"][0]:
            if metadata and "text" in metadata:
                context_pieces.append(metadata["text"])
        context = "\n".join(context_pieces)
    
    if not context.strip():
        context = "No relevant context found in the knowledge base."

    # Create a better prompt template
    if "phi" in selected_model_path.lower():
        # Phi-3.5 format
        prompt = f"""<|system|>
You are a helpful AI assistant. Answer questions concisely and accurately based on the provided context. If the context doesn't contain relevant information, say so clearly.

<|user|>
Context: {context}

Question: {question}

<|assistant|>
"""
    elif "qwen" in selected_model_path.lower():
        # Qwen2 format
        prompt = f"""<|im_start|>system
You are a helpful AI assistant. Answer questions concisely and accurately based on the provided context.<|im_end|>
<|im_start|>user
Context: {context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""
    elif "llama" in selected_model_path.lower():
        # Llama format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant. Answer questions concisely and accurately based on the provided context.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Context: {context}

Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    else:
        # Generic format for TinyLlama and others
        prompt = f"""### System:
You are a helpful AI assistant. Answer questions concisely and accurately based on the provided context.

### Context:
{context}

### Question:
{question}

### Answer:
"""

    try:
        # Generate response
        response = pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=pipe.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        generated_text = response[0]["generated_text"]
        
        # Clean up the response
        if isinstance(generated_text, str):
            # Remove prompt remnants and clean up
            lines = generated_text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('###', '<|', 'Context:', 'Question:')):
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                return ' '.join(cleaned_lines[:3])  # Take first few sentences
            else:
                return generated_text
        
        return generated_text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# -------------------------------
# Streamlit UI with Enhanced Features
# -------------------------------
st.title("🤖 Enhanced RAG Chatbot - Developed by Rayyan Ahmed")
st.write(f"Currently using: **{selected_model_name}** | Ask me anything from your knowledge base!")

# Display model info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", selected_model_name.split()[0])
with col2:
    st.metric("Device", device.upper())
with col3:
    if hasattr(collection, '_count') or hasattr(collection, 'count'):
        try:
            doc_count = collection.count()
            st.metric("Documents", doc_count)
        except:
            st.metric("Documents", "Unknown")
    else:
        st.metric("Documents", "Ready")

# Settings expander
with st.expander("⚙️ Advanced Settings"):
    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider("Max Response Length", 50, 200, 100)
        temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.1, 0.1)
    with col2:
        top_p = st.slider("Top-p (Focus)", 0.1, 1.0, 0.9, 0.1)

# Chat interface
user_query = st.text_input("💬 Enter your question:", placeholder="What would you like to know?")

col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("🚀 Ask", type="primary")
with col2:
    clear_button = st.button("🗑️ Clear")

if ask_button and user_query.strip():
    with st.spinner("🔍 Searching knowledge base and generating response..."):
        answer = rag_query(
            user_query, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p
        )
    
    st.success("✅ Answer:")
    st.write(answer)
    
    # Show context used (optional)
    with st.expander("📚 View Retrieved Context"):
        results = collection.query(query_texts=[user_query], n_results=3)
        if results and results.get("metadatas") and results["metadatas"][0]:
            for i, metadata in enumerate(results["metadatas"][0], 1):
                if metadata and "text" in metadata:
                    st.write(f"**Context {i}:** {metadata['text'][:200]}...")

elif ask_button:
    st.warning("⚠️ Please enter a question.")

if clear_button:
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 12px;'>"
    "RAG Chatbot v2.0 | Enhanced with Modern Language Models | Built by Rayyan Ahmed"
    "</div>", 
    unsafe_allow_html=True
)