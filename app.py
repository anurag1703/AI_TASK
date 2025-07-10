import os
import time
import streamlit as st
import tempfile
from langchain_groq.chat_models import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

# Set HuggingFace token from Streamlit secrets
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

# Define embedding model
embedding_model = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

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
st.title("📘 RAG Chatbot with PDF Upload + Streaming via Groq")

# Sidebar: Groq key and info
with st.sidebar:
    st.subheader("🔑 API Key Setup")
    groq_key = st.text_input("Enter your Groq API Key", type="password")

    st.subheader("ℹ️ System Info")
    st.write("Model: `LLaMA3-8B` via Groq")
    st.write(f"Embedding Model: `{embedding_model}`")
    if "chunks" in st.session_state:
        st.write(f"Chunks Loaded: {len(st.session_state.chunks)}")
    else:
        st.write("Chunks Loaded: ❌")

    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# Upload PDF file
uploaded_file = st.file_uploader("📄 Upload a PDF document", type=["pdf"])

# Vector embedding logic
def prepare_vector_db_from_file(uploaded_file):
    try:
        if uploaded_file is None:
            st.warning("⚠️ Please upload a PDF file first.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()

        splitter = SentenceTransformersTokenTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = splitter.split_documents(docs)

        vectordb = FAISS.from_documents(chunks, embeddings)
        st.session_state.chunks = chunks
        st.session_state.vectordb = vectordb
        st.success(f"✅ Document processed and {len(chunks)} chunks embedded.")

        os.remove(tmp_path)  # clean up temp file
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")

# Display streaming token-by-token
def stream_response(response):
    placeholder = st.empty()
    collected = ""
    for chunk in response:
        token = chunk.get("answer", "") if isinstance(chunk, dict) else ""
        collected += token
        placeholder.markdown(collected)
    return collected

# Only proceed if Groq API key is entered
if groq_key:
    try:
        os.environ["GROQ_API_KEY"] = groq_key
        llm = ChatGroq(groq_api_key=groq_key, model_name="Llama3-8b-8192", streaming=True)
        llm.model_rebuild()

        if st.button("📥 Load and Embed Uploaded Document"):
            prepare_vector_db_from_file(uploaded_file)

        user_query = st.chat_input("Ask a question about the uploaded PDF...")

        if user_query and "vectordb" in st.session_state:
            retriever = st.session_state.vectordb.as_retriever()
            document_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, document_chain)

            with st.spinner("💬 Generating response..."):
                start = time.time()
                response = rag_chain.stream({"input": user_query})
                final_output = stream_response(response)
                st.caption(f"⏱️ Response time: {round(time.time() - start, 2)} seconds")

            with st.expander("📄 Retrieved Chunks"):
                matched_docs = retriever.invoke(user_query)
                for i, doc in enumerate(matched_docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.markdown(doc.page_content)
                    st.markdown("---")
        elif user_query:
            st.warning("📥 Please upload and embed a document first.")
    except Exception as err:
        st.error(f"❌ Error initializing chatbot: {err}")
else:
    st.info("🔐 Please enter your Groq API key in the sidebar to start.")

