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
```

2️⃣ Create Virtual Environment
      
 - python -m venv venv

Activate it:

Windows:

 - venv\Scripts\activate

macOS/Linux:

 - source venv/bin/activate

3️⃣ Install Dependencies

 - pip install -r requirements.txt

🔐 Environment Variables Setup :

Create a file named .env in the project root directory and add:

GROQ_API_KEY=your_api_key_here

Make sure .env is included in .gitignore to keep your API key secure.

▶️ Run the Application
- python app.py

Open your browser and visit:

  http://127.0.0.1:5000

  
🌍 Deployment

This application can be deployed on:

Render

Railway

Replit

Any VPS using Gunicorn


📁 Project Structure

DocuMind-AI-Intelligent-PDF-Chat-Assistant/
│

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

├── templates/

│   └── index.html

├── static/

├── uploads/ (ignored)

├── kb.index (ignored)

├── kb.pkl (ignored)

└── .env (ignored)

📌 Future Enhancements

Persistent vector database (e.g., Pinecone, Weaviate)

User authentication system

Multi-document support

Improved UI/UX design

Streaming LLM responses

Cloud storage integration
