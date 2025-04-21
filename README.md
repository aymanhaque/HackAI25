# 📊 Prism — AI-Powered Financial Report Advisor

FinSight is an AI-powered assistant designed for investors, analysts, and shareholders. It leverages Retrieval-Augmented Generation (RAG), Google Gemini, LangChain, and ChromaDB to make financial reports accessible, understandable, and actionable.

## 🚀 Features

- 🔍 **RAG-based Question Answering**: Ask complex questions about lengthy financial reports and get precise, context-aware answers.
- 📚 **PDF Document Parsing**: Upload annual reports in PDF format — the app automatically extracts and indexes content.
- 🧠 **LangChain Integration**: Orchestrates the entire pipeline of document loading, chunking, embedding, and querying.
- 🤖 **Google Gemini LLM**: Generates high-quality, insightful responses in natural language.
- 🗂️ **ChromaDB Vector Store**: Efficiently stores and retrieves embedded document chunks for fast contextual answers.

## 🛠️ Tech Stack

- `LangChain` for chaining and orchestration
- `Google Gemini` for LLM-powered reasoning
- `ChromaDB` for vector storage and retrieval
- `PyMuPDF` for PDF parsing
- `HuggingFace Embeddings` for document chunk vectorization

## 📦 Installation

1. **Clone the repo**
2. ```bash
   git clone https://github.com/aymanhaque/HackAI25.git
   cd HackAI25
   pip install -r requirements.txt
   python3 backend/new.py
3. On a different terminal:
4. ```bash
   cd HackAI25
   cd frontend/my-app
   npm install
   npm run dev
