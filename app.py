from flask import Flask, request, render_template, session
import os
from dotenv import load_dotenv
load_dotenv()

import pickle
import numpy as np
import pdfplumber
import re
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ---------------- #

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "chatbot_secret_key_2026"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Lazy load embedding model
embedding_model = None


def get_model():
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model


# ---------------- PDF FUNCTIONS ---------------- #

import pdfplumber
import re

def read_pdf(path):

    text = ""
    links = []

    with pdfplumber.open(path) as pdf:

        for page in pdf.pages:

            # Extract text
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

            # Extract hyperlinks
            if page.hyperlinks:
                for link in page.hyperlinks:
                    if "uri" in link:
                        links.append(link["uri"])

    # Append hyperlinks to text so the model can see them
    if links:
        text += "\n\nLinks found in document:\n"
        for l in links:
            text += l + "\n"

    # Clean formatting
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("—", " ")
    text = text.replace("–", " ")

    return text

# Better chunking (keeps sentences intact)
def chunk_text(text, size=500):

    paragraphs = text.split("\n")

    chunks = []
    current = ""

    for p in paragraphs:

        if len(current) + len(p) < size:
            current += p + "\n"
        else:
            chunks.append(current)
            current = p + "\n"

    if current:
        chunks.append(current)

    return chunks


# ---------------- ROUTES ---------------- #

@app.route("/")
def home():

    if "chat" not in session:
        session["chat"] = []

    return render_template(
        "index.html",
        chat=session["chat"],
        filename=session.get("filename")
    )


@app.route("/reset")
def reset():

    session.clear()

    return render_template("index.html", chat=[])


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("pdf")

    if not file or file.filename == "":
        return render_template("index.html", msg="❌ No file selected")

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    raw = read_pdf(path)

    if not raw.strip():
        return render_template("index.html", msg="❌ PDF has no readable text")

    texts = chunk_text(raw)

    # Generate embeddings
    model = get_model()
    embeddings = model.encode(texts).astype("float32")

    # Save knowledge base
    with open("kb.pkl", "wb") as f:
        pickle.dump(texts, f)

    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    session["filename"] = file.filename
    session["chat"] = []

    return render_template(
        "index.html",
        msg="✅ PDF uploaded!",
        filename=file.filename,
        chat=[]
    )


@app.route("/ask", methods=["POST"])
def ask():

    question = request.form.get("question")

    if not question:
        return render_template("index.html", msg="❌ Please enter a question")

    # Load stored data
    with open("kb.pkl", "rb") as f:
        texts = pickle.load(f)

    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    # Embed question
    model = get_model()
    q_emb = model.encode([question]).astype("float32")

    # Similarity search
    similarities = cosine_similarity(q_emb, embeddings)[0]

    top_indices = similarities.argsort()[-5:][::-1]

    context = "\n".join([texts[i] for i in top_indices])

    prompt = f"""
You are a document assistant.

STRICT RULES:
- Answer ONLY using the provided context
- Do NOT modify emails, URLs, numbers, or names
- Do NOT guess missing information
- If the answer is not present in the context, say:
"This information is not available in the uploaded PDF."

Use bullet points when helpful.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    answer = response.choices[0].message.content

    if "chat" not in session:
        session["chat"] = []

    session["chat"].append({"role": "user", "msg": question})
    session["chat"].append({"role": "bot", "msg": answer})

    session.modified = True

    return render_template(
        "index.html",
        chat=session["chat"],
        filename=session.get("filename")
    )


# ---------------- RUN SERVER ---------------- #

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)