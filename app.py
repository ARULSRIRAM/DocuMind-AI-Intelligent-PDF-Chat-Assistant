from flask import Flask, request, render_template, session
import os
from dotenv import load_dotenv
load_dotenv()
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from groq import Groq


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "chatbot_secret_key_2026"

model = SentenceTransformer("all-MiniLM-L6-v2") 
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))# Replace with your actual key

texts = []
index = None

# -------- PDF FUNCTIONS -------- #

def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# -------- ROUTES -------- #

@app.route("/")
def home():
    if "chat" not in session:
        session["chat"] = []
    return render_template("index.html", chat=session["chat"], filename=session.get("filename"))

@app.route("/reset")
def reset():
    session.clear()
    return render_template("index.html", chat=[])

@app.route("/upload", methods=["POST"])
def upload():
    global texts, index

    file = request.files["pdf"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    raw = read_pdf(path)
    texts = chunk_text(raw)

    with open("kb.pkl", "wb") as f:
        pickle.dump(texts, f)

    emb = model.encode(texts)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(emb).astype("float32"))
    faiss.write_index(index, "kb.index")

    session["filename"] = file.filename
    session["chat"] = []

    return render_template("index.html", msg="✅ PDF uploaded!", filename=file.filename, chat=[])

@app.route("/ask", methods=["POST"])
def ask():
    global texts, index

    question = request.form["question"]

    # Load knowledge base
    with open("kb.pkl", "rb") as f:
        texts = pickle.load(f)
    index = faiss.read_index("kb.index")

    # Search PDF
    q_emb = model.encode([question])
    _, I = index.search(np.array(q_emb).astype("float32"), k=3)
    context = "\n".join([texts[i] for i in I[0]])

    prompt = f"""
You are a helpful document assistant. Answer the question based only on the provided context.
- Use bullet points or numbered lists when listing items or steps.
- Use **bold** for key terms.
- Keep your answer clear and well-structured.
- If the answer is not in the context, say: "This information is not available in the uploaded PDF."

Context:
{context}

Question: {question}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    answer = response.choices[0].message.content

    # Save chat history
    if "chat" not in session:
        session["chat"] = []

    session["chat"].append({"role": "user", "msg": question})
    session["chat"].append({"role": "bot", "msg": answer})
    session.modified = True

    return render_template("index.html", chat=session["chat"], filename=session.get("filename"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)