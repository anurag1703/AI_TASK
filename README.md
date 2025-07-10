# 🧠 RAG Chatbot with PDF Upload + Streaming (LLaMA3 via Groq)

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built using:
- ✅ PDF Upload Interface
- ✅ Sentence-aware document chunking
- ✅ Embedding with HuggingFace MiniLM
- ✅ FAISS vector database
- ✅ Real-time response streaming via Groq-hosted **LLaMA3**
- ✅ Modern UI with Streamlit

---

## 🚀 Features

- 📄 Upload any PDF document
- 🔍 Chunk using sentence-aware splitting (`RecursiveCharacterTextSplitter`)
- 🧠 Embed with `all-MiniLM-L6-v2`
- 🔎 Store in FAISS vector DB (in-memory)
- 💬 Query the document using LLM-based answer generation
- ⏱️ Streamed token-wise output for responsiveness
- 🔗 Show retrieved document chunks for transparency
- 🧑‍💻 Each user enters their own **Groq API Key** to protect usage

---

## 📁 Folder Structure


├── app.py
├── .env 
├── requirements.txt
└── README.md 

## Future Enhancements
Multi-PDF document support

Persistent FAISS vector store

Chat history or threaded UI

Authentication and user tracking

Export Q&A session as PDF


## 🧾 License
This project is MIT licensed.
You're free to fork, use, and modify for your own work.

## 🔗 Deployment & Demo
📂 GitHub Repository: [Insert your repo link here]

🌐 Streamlit App: [Insert your Streamlit Cloud deployment link here]