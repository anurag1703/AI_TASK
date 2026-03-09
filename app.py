import os
import time
import streamlit as st
import tempfile

# --- LATEST IMPORTS (LangChain v1.0+) ---
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Note: Retrieval chains moved to langchain_classic in 2025/2026 updates
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# Set HuggingFace token from Streamlit secrets
if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

# Define embedding model
# all-MiniLM-L6-v2 has a max token limit of 384. 
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Define prompt template
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant. Use ONLY the context below to answer the user's question.
If the answer is not in the context, say: "The information is not available in the document."

<context>
{context}
</context>

Question: {input}
""")

# Streamlit UI setup
st.set_page_config(page_title="PDF Q&A Chatbot", layout="wide")
st.title("📘 RAG Chatbot (LangChain v1.0 / Groq)")

# Sidebar
with st.sidebar:
    st.subheader("🔑 API Key Setup")
    groq_key = st.text_input("Enter your Groq API Key", type="password")

    st.subheader("ℹ️ System Info")
    st.write("Model: `LLaMA3-8B` via Groq")
    st.write(f"Embedding Model: `{embedding_model_name}`")
    
    if "chunks" in st.session_state:
        st.write(f"Chunks Loaded: {len(st.session_state.chunks)}")
    else:
        st.write("Chunks Loaded: ❌")

    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# Upload PDF file
uploaded_file = st.file_uploader("📄 Upload a PDF document", type=["pdf"])

def prepare_vector_db_from_file(file):
    try:
        if file is None:
            st.warning("⚠️ Please upload a PDF file first.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue()) # Use getvalue() for uploaded file stream
            tmp_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()

        # UPDATED: Adjusted chunk_size to 384 to match all-MiniLM-L6-v2's max sequence length
        splitter = SentenceTransformersTokenTextSplitter(chunk_size=384, chunk_overlap=40)
        chunks = splitter.split_documents(docs)

        vectordb = FAISS.from_documents(chunks, embeddings)
        st.session_state.chunks = chunks
        st.session_state.vectordb = vectordb
        st.success(f"✅ Document processed: {len(chunks)} chunks embedded.")

        os.remove(tmp_path) 
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")

def stream_response(response_iterator):
    placeholder = st.empty()
    collected = ""
    for chunk in response_iterator:
        # LangChain stream returns dicts; 'answer' contains the text tokens
        token = chunk.get("answer", "")
        collected += token
        placeholder.markdown(collected)
    return collected

if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
    llm = ChatGroq(
        model="Llama3-8b-8192",
        temperature=0,
        streaming=True
    )

    if st.button("📥 Load and Embed Uploaded Document"):
        prepare_vector_db_from_file(uploaded_file)

    user_query = st.chat_input("Ask a question about the uploaded PDF...")

    if user_query and "vectordb" in st.session_state:
        retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 3})
        
        # Build the chain using classic chain helpers
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("💬 Generating response..."):
            start = time.time()
            # In 1.0, stream() is the preferred way to handle real-time output
            response = rag_chain.stream({"input": user_query})
            final_output = stream_response(response)
            st.caption(f"⏱️ Response time: {round(time.time() - start, 2)} seconds")

        with st.expander("📄 Retrieved Chunks"):
            matched_docs = retriever.invoke(user_query)
            for i, doc in enumerate(matched_docs):
                st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}\n---")
    elif user_query:
        st.warning("📥 Please upload and embed a document first.")
else:
    st.info("🔐 Please enter your Groq API key in the sidebar to start.")
