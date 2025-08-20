import streamlit as st
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

import os
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Background Image
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), url("https://static.vecteezy.com/system/resources/previews/020/067/380/non_2x/chat-ai-conversation-method-background-free-vector.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Import libraries with error handling
try:
    from transformers import pipeline
    import torch
    import chromadb
    DEPENDENCIES_LOADED = True
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.stop()

# -------------------------------
# Sidebar Info
# -------------------------------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: rgba(0, 50, 80, 0.7);
        color: white;
    }
    [data-testid="stSidebar"] .css-1v3fvcr {
        color: white;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #00cfff;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📁 Project Info")
    st.write("RAG Chatbot using modern instruction-tuned models for better responses.")
    
    st.markdown("### 👨‍💻 Developer")
    st.write("**Rayyan Ahmed**")
    st.write("- IBM Certified Advanced LLM FineTuner")
    st.write("- Google Certified Soft Skills Professional")
    st.write("- Hugging Face Certified in LLM Fundamentals")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")
    
    st.markdown("### 🛠️ Tech Stack")
    st.write("- **Python 3** & **Streamlit**")
    st.write("- **Transformers** (Hugging Face)")
    st.write("- **PyTorch** (Model Backend)")
    st.write("- **ChromaDB** (Vector Database)")
    st.write("- **RAG Architecture**")

# -------------------------------
# Check device availability
# -------------------------------
@st.cache_data
def get_device_info():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    else:
        return "cpu", "CPU"

device, device_name = get_device_info()

with st.sidebar:
    if device == "cuda":
        st.success(f"✅ GPU: {device_name}")
    else:
        st.info("💻 Running on CPU")

# -------------------------------
# Simple Model Options for Streamlit Cloud
# -------------------------------
MODEL_OPTIONS = {
    "DistilGPT-2 (Fast & Light)": "distilgpt2",
    "GPT-2 Small": "gpt2",
    "TinyLlama Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}

with st.sidebar:
    st.markdown("### 🎯 Model Selection")
    selected_model_name = st.selectbox(
        "Choose Model:",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="Lighter models work better on Streamlit Cloud"
    )
    selected_model_path = MODEL_OPTIONS[selected_model_name]

# -------------------------------
# Load Model with Streamlit Cloud optimizations
# -------------------------------
@st.cache_resource
def load_model(model_name):
    try:
        with st.spinner(f"Loading {model_name}..."):
            if "distil" in model_name.lower():
                pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    device=-1,  # Force CPU for Streamlit Cloud
                    return_full_text=False,
                    pad_token_id=50256
                )
            else:
                pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    device=-1,  # Force CPU
                    return_full_text=False
                )
            return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Fallback to smallest model
        return pipeline("text-generation", model="distilgpt2", device=-1, return_full_text=False)

# -------------------------------
# Initialize ChromaDB with error handling
# -------------------------------
@st.cache_resource
def init_chromadb():
    try:
        client = chromadb.Client()  # Use in-memory client for Streamlit Cloud
        collection = client.get_or_create_collection("rag_collection")
        
        # Add some sample documents if collection is empty
        if collection.count() == 0:
            sample_docs = [
                "Artificial Intelligence is transforming various industries.",
                "Machine Learning is a subset of AI that focuses on learning from data.",
                "Natural Language Processing helps computers understand human language.",
                "Deep Learning uses neural networks with multiple layers.",
                "Python is a popular programming language for AI development."
            ]
            collection.add(
                documents=sample_docs,
                metadatas=[{"text": doc, "source": "sample"} for doc in sample_docs],
                ids=[f"doc_{i}" for i in range(len(sample_docs))]
            )
        
        return collection
    except Exception as e:
        st.error(f"ChromaDB initialization error: {e}")
        return None

# Load model and database
pipe = load_model(selected_model_path)
collection = init_chromadb()

# -------------------------------
# RAG Query Function (Simplified for Streamlit Cloud)
# -------------------------------
def rag_query(question, max_tokens=80):
    if not collection:
        return "Database not available. Please check the setup."
    
    try:
        # Retrieve context
        results = collection.query(query_texts=[question], n_results=3)
        context = ""
        
        if results and results.get("documents") and results["documents"][0]:
            context = " ".join(results["documents"][0])
        
        if not context.strip():
            context = "No relevant context found."
        
        # Create simple prompt
        prompt = f"""Answer this question based on the context provided.

Context: {context}

Question: {question}

Answer:"""
        
        # Generate response
        response = pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id if hasattr(pipe.tokenizer, 'eos_token_id') else 50256
        )
        
        if response and len(response) > 0:
            answer = response[0]["generated_text"].strip()
            # Clean up the answer
            if len(answer) > 200:
                sentences = answer.split('.')
                if len(sentences) > 1:
                    answer = '. '.join(sentences[:2]) + '.'
            return answer
        else:
            return "Sorry, I couldn't generate a response."
            
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🤖 RAG Chatbot - Developed by Rayyan Ahmed")
st.markdown(f"**Current Model:** {selected_model_name}")

# Display status
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Status", "✅ Ready" if pipe and collection else "❌ Error")
with col2:
    st.metric("Device", device.upper())
with col3:
    if collection:
        doc_count = collection.count()
        st.metric("Documents", doc_count)
    else:
        st.metric("Documents", "N/A")

st.write("Ask me anything! I'll search the knowledge base and generate an answer.")

# Chat interface
user_query = st.text_input(
    "💬 Enter your question:",
    placeholder="What would you like to know about AI, ML, or programming?",
    key="user_input"
)

col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("🚀 Ask", type="primary")
with col2:
    if st.button("🔄 Reset"):
        st.rerun()

if ask_button and user_query.strip():
    with st.spinner("🤔 Thinking..."):
        try:
            answer = rag_query(user_query)
            st.success("✅ Answer:")
            st.write(answer)
            
            # Show retrieved context
            if collection:
                with st.expander("📚 Retrieved Context"):
                    results = collection.query(query_texts=[user_query], n_results=3)
                    if results and results.get("documents"):
                        for i, doc in enumerate(results["documents"][0], 1):
                            st.write(f"**{i}.** {doc}")
                            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
elif ask_button:
    st.warning("⚠️ Please enter a question.")

# Add some example questions
with st.expander("💡 Try these example questions"):
    examples = [
        "What is Artificial Intelligence?",
        "How does Machine Learning work?",
        "What is Natural Language Processing?",
        "Explain Deep Learning",
        "Why is Python popular for AI?"
    ]
    
    for example in examples:
        if st.button(example, key=f"example_{example[:10]}"):
            st.experimental_set_query_params(q=example)
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "🤖 RAG Chatbot v2.0 | Streamlit Cloud Compatible | Built by Rayyan Ahmed"
    "</div>",
    unsafe_allow_html=True
)
