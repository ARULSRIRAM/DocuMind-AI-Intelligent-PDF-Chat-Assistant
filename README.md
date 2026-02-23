# 📄 DocuMind AI – Intelligent PDF Chat Assistant

DocuMind AI is a Retrieval-Augmented Generation (RAG) based PDF chatbot that allows users to upload PDF documents and interact with them using natural language queries. The system performs semantic search over document embeddings and generates intelligent, context-aware responses.

---

## 🚀 Features

- 📂 Upload any PDF document
- 🔎 Semantic search using FAISS
- 🧠 Embedding generation with Sentence-Transformers
- 🤖 Context-aware responses using Groq LLM
- 💬 Chat-style interaction interface
- 🔐 Secure API key handling using environment variables
- 🌐 Deployment-ready Flask application

---

## 🏗️ Architecture

1. PDF is uploaded and text is extracted using PyPDF2  
2. Text is split into chunks  
3. Sentence-Transformers converts chunks into vector embeddings  
4. FAISS indexes embeddings for fast similarity search  
5. User query is converted into embedding  
6. Top relevant chunks are retrieved  
7. Groq LLM generates response based on retrieved context  

This follows the **Retrieval-Augmented Generation (RAG)** approach.

---

## 🛠️ Tech Stack

- Python
- Flask
- FAISS
- Sentence-Transformers
- Groq LLM
- NumPy
- PyPDF2

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ARULSRIRAM/DocuMind-AI-Intelligent-PDF-Chat-Assistant.git
cd DocuMind-AI-Intelligent-PDF-Chat-Assistant
