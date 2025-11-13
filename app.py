from flask import Flask, render_template, request, send_file
import os
import re
import pdfplumber
import docx2txt
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Flask setup
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

excel_file = "resume_ranking.xlsx"

# -------------------------------
# Helper Functions
# -------------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

def extract_text(file_path):
    text = ""
    try:
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        elif file_path.endswith(".docx"):
            text = docx2txt.process(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        uploaded_files = request.files.getlist("resumes")
        resumes_texts = []
        resume_files = []

        for file in uploaded_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            text = extract_text(file_path)
            cleaned_text = clean_text(text)
            resumes_texts.append(cleaned_text)
            resume_files.append(file.filename)

        # Job Description
        jd_text = clean_text(request.form["job_description"])

        # TF-IDF Similarity
        vectorizer = TfidfVectorizer(max_features=5000)
        all_texts = resumes_texts + [jd_text]
        X = vectorizer.fit_transform(all_texts)
        resume_vectors = X[:-1]
        jd_vector = X[-1]
        tfidf_scores = cosine_similarity(resume_vectors, jd_vector).flatten()

        # BERT Embeddings Similarity
        model = SentenceTransformer('all-MiniLM-L6-v2')
        all_texts_raw = [extract_text(os.path.join(app.config['UPLOAD_FOLDER'], f)) for f in resume_files] + [request.form["job_description"]]
        embeddings = model.encode(all_texts_raw)
        bert_scores = [cosine_sim(emb, embeddings[-1]) for emb in embeddings[:-1]]

        # Rank resumes by BERT
        ranked = sorted(zip(resume_files, tfidf_scores, bert_scores), key=lambda x: x[2], reverse=True)

        # Save Excel
        df = pd.DataFrame(ranked, columns=["Resume", "TF-IDF Score", "BERT Score"])
        df.to_excel(excel_file, index=False)

        results = ranked

    return render_template("index.html", results=results)

@app.route("/download")
def download():
    if os.path.exists(excel_file):
        return send_file(excel_file, as_attachment=True)
    else:
        return "No file found!"

if __name__ == "__main__":
    app.run(debug=True)
