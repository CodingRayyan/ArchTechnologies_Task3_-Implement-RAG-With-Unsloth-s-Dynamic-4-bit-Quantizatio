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


from transformers import pipeline
import chromadb
import torch

# -------------------------------
# Sidebar Info
# -------------------------------
# -------------------------------
# Sidebar Info for RAG Chatbot
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
        "- **This is a RAG Chatbot using a local instruction-tuned model.**\n"
        "  It searches your knowledge base and generates answers using GPT-Neo."
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
    st.markdown("- **Transformers / Hugging Face** (for GPT-Neo and other LLMs, e.g., EleutherAI, Unsloth, Ollama)")
    st.markdown("- **Torch / PyTorch** (for model backend, GPU support, 4-bit quantization)")
    st.markdown("- **ChromaDB** (for RAG retrieval and document indexing)")
    st.markdown("- **Streamlit** (for web app interface)")
    st.markdown("- **Numpy** (numerical operations)")
    st.markdown("- **Pandas** (data handling and preprocessing)")
    st.markdown("- **Pickle** (for storing scalers or objects)")
    st.markdown("- **Matplotlib / Seaborn** (optional visualization)")
    st.markdown("- **Optional:** TensorFlow, Keras (if using other ML/ANN models)")
    st.markdown("- **RAG Architecture** (Retrieval-Augmented Generation setup integrating ChromaDB + LLM)")

# -------------------------------
# Check GPU availability
# -------------------------------
if torch.cuda.is_available():
    print("GPU is available ✅")
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("GPU not available ❌, using CPU")

# -------------------------------
# Load instruction-tuned model
# -------------------------------
from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model():
    return pipeline(
        task="text2text-generation",      # correct task for T5
        model="google/t5-small-ssm",     # full HF model name
        device=-1,                        # CPU
        torch_dtype="auto"
    )

pipe = load_model()

# -------------------------------
# ChromaDB Setup
# -------------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_collection")

# -------------------------------
# RAG Query Function
# -------------------------------
# -------------------------------
# RAG Query Function (Concise Answer)
# -------------------------------
def rag_query(question, max_tokens=50, do_sample=False):
    results = collection.query(query_texts=[question], n_results=3)
    context = ""
    if results and results.get("metadatas") and results["metadatas"][0]:
        context = " ".join(m.get("text", "") for m in results["metadatas"][0] if m)
    if not context.strip():
        context = "No context available."

    prompt = (
        f"Answer the following question concisely in English in one sentence.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    response = pipe(prompt, max_new_tokens=max_tokens, do_sample=do_sample)[0]["generated_text"]
    return response

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🤖 RAG Chatbot with Instruction-Tuned Model - Developed by Rayyan Ahmed")
st.write("Ask me anything, I’ll search your knowledge base + use an instruction-tuned model for answers.")

# Chat input
user_query = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            answer = rag_query(user_query)  # uses max_tokens=30 by default
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
