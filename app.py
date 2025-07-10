import os
import time
import streamlit as st
from io import BytesIO
from langchain_groq.chat_models import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

# Setup HF token from st.secrets (required for Streamlit Cloud)
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

# Initialize embedding model
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

# UI setup
st.set_page_config(page_title="PDF Q&A Chatbot", layout="wide")
st.title("📘 RAG Chatbot - PDF Upload + Groq + LLaMA3")

# Sidebar: API key + metadata
with st.sidebar:
    st.subheader("🔑 API Key Setup")
    groq_key = st.text_input("Enter your Groq API Key", type="password")

    st.subheader("ℹ️ System Info")
    st.write("Model: `LLaMA3-8B` via Groq")
    st.write(f"Embedding: `{embedding_model}`")

    if "chunks" in st.session_state:
        st.write(f"Chunks Loaded: {len(st.session_state.chunks)}")
    else:
        st.write("Chunks Loaded: ❌")

    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# File uploader
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

# Vector DB creation
def prepare_vector_db_from_file(uploaded_file):
    try:
        if uploaded_file is None:
            st.warning("⚠️ Please upload a PDF first.")
            return

        file_bytes = uploaded_file.read()
        temp_io = BytesIO(file_bytes)

        loader = PyPDFLoader(temp_io)
        docs = loader.load()

        splitter = SentenceTransformersTokenTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = splitter.split_documents(docs)

        vectordb = FAISS.from_documents(chunks, embeddings)
        st.session_state.chunks = chunks
        st.session_state.vectordb = vectordb
        st.success(f"✅ Loaded {len(chunks)} chunks from uploaded PDF.")
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")

# Stream response display
def stream_response(response):
    placeholder = st.empty()
    collected = ""
    for chunk in response:
        token = chunk.get("answer", "") if isinstance(chunk, dict) else ""
        collected += token
        placeholder.markdown(collected)
    return collected

# Initialize and run app logic only if API key is provided
if groq_key:
    try:
        os.environ["GROQ_API_KEY"] = groq_key
        llm = ChatGroq(groq_api_key=groq_key, model_name="Llama3-8b-8192", streaming=True)
        llm.model_rebuild()

        if st.button("📥 Load and Embed Uploaded Document"):
            prepare_vector_db_from_file(uploaded_file)

        user_query = st.chat_input("Ask a question from the uploaded PDF...")

        if user_query and "vectordb" in st.session_state:
            retriever = st.session_state.vectordb.as_retriever()
            document_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, document_chain)

            with st.spinner("Generating response..."):
                start = time.time()
                response = rag_chain.stream({"input": user_query})
                final_output = stream_response(response)
                duration = time.time() - start

            st.caption(f"⏱️ Response time: {round(duration, 2)} seconds")

            with st.expander("📄 Retrieved Chunks"):
                matched_docs = retriever.invoke(user_query)
                for i, doc in enumerate(matched_docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.markdown(doc.page_content)
                    st.markdown("---")
        elif user_query:
            st.warning("📥 Please load and embed the uploaded document first.")
    except Exception as err:
        st.error(f"❌ Error initializing LLM: {err}")
else:
    st.info("🔐 Please enter your Groq API Key in the sidebar to activate the chatbot.")

