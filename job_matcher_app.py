import streamlit as st
import os
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def load_resumes(folder_path):
    resumes = {}
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(folder_path, file))
            resumes[file] = text
    return resumes

def load_jobs(folder_path):
    jobs = {}
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                jobs[file] = f.read()
    return jobs

def match_resumes(resumes, jobs):
    results = []
    for res_name, res_text in resumes.items():
        res_vec = model.encode([res_text])[0]
        for job_name, job_text in jobs.items():
            job_vec = model.encode([job_text])[0]
            score = cosine_similarity([res_vec], [job_vec])[0][0]
            results.append((res_name, job_name, round(score * 100, 2)))
    return sorted(results, key=lambda x: x[2], reverse=True)

st.title("🧠 Lebenslauf ↔ Stellenanzeige Matching")

st.markdown("📂 PDFs in `lebenslaeufe/` und Textdateien in `stellenanzeigen/` hochladen.")

if st.button("🔍 Matching starten"):
    resumes = load_resumes("lebenslaeufe")
    jobs = load_jobs("stellenanzeigen")
    if not resumes:
        st.warning("❗ Keine Lebensläufe gefunden.")
    elif not jobs:
        st.warning("❗ Keine Stellenanzeigen gefunden.")
    else:
        matches = match_resumes(resumes, jobs)
        st.success(f"✅ {len(matches)} Matches berechnet.")
        st.table(matches)
