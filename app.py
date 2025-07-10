import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load HF token from .env
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Set up embeddings
embedding_model = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant. Use ONLY the context below to answer the user's question.
If the answer is not in the context, say: "The information is not available in the document."

<context>
{context}
</context>

Question: {input}
""")

# Set up Streamlit page
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📘 RAG Chatbot with PDF Upload and Streaming via Groq")

# Sidebar
with st.sidebar:
    st.subheader("🔑 API Key Setup")
    groq_key = st.text_input("Enter your Groq API Key", type="password")

    st.subheader("ℹ️ System Info")
    st.write("Model: `LLaMA3-8B` via Groq")
    st.write(f"Embedding: `{embedding_model}`")
    if "chunks" in st.session_state:
        st.write(f"Chunks: {len(st.session_state.chunks)}")
    else:
        st.write("Chunks: ❌ Not loaded")

    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# PDF Upload
uploaded_file = st.file_uploader("📄 Upload a PDF document", type=["pdf"])

# Vector DB creation
def prepare_vector_db_from_file(uploaded_file):
    if uploaded_file is not None:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp_uploaded.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        chunks = splitter.split_documents(docs)
        vectordb = FAISS.from_documents(chunks, embeddings)
        st.session_state.chunks = chunks
        st.session_state.vectordb = vectordb
        st.success(f"✅ Vector DB created with {len(chunks)} chunks.")
    else:
        st.warning("⚠️ Please upload a PDF file first.")

# Stream response
def stream_response(response):
    placeholder = st.empty()
    collected = ""
    for chunk in response:
        token = chunk.get("answer", "") if isinstance(chunk, dict) else ""
        collected += token
        placeholder.markdown(collected)
    return collected

# Load LLM only if Groq API key is provided
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
    llm = ChatGroq(groq_api_key=groq_key, model_name="Llama3-8b-8192", streaming=True)
    llm.model_rebuild()

    # Button to embed PDF
    if st.button("📥 Load and Embed Uploaded Document"):
        prepare_vector_db_from_file(uploaded_file)

    # Chat input
    user_query = st.chat_input("Ask a question about the uploaded PDF...")

    if user_query and "vectordb" in st.session_state:
        retriever = st.session_state.vectordb.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Generating response..."):
            start = time.time()
            response = rag_chain.stream({"input": user_query})
            final_output = stream_response(response)
            duration = time.time() - start

        st.caption(f"⏱️ Time taken: {round(duration, 2)} seconds")

        with st.expander("📄 Retrieved Document Chunks"):
            matched_docs = retriever.invoke(user_query)
            for i, doc in enumerate(matched_docs):
                st.markdown(f"**Chunk {i+1}:**")
                st.markdown(doc.page_content)
                st.markdown("---")

    elif user_query:
        st.warning("⚠️ Please upload and embed a document first.")
else:
    st.warning("🔐 Please enter your Groq API key in the sidebar to activate the chatbot.")
