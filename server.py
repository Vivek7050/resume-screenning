import os
# Suppress TensorFlow/oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sqlite3
import re
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import spacy
import torch
import numpy as np
from pdfminer.high_level import extract_text
# Yahan BertTokenizer ki jagah AutoTokenizer aur AutoModel lagaya hai
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk

# ==========================================
# 1. INITIALIZE FASTAPI & NLP MODELS
# ==========================================
app = FastAPI(title="AI Resume Screener & Builder Engine")

# Allow Android app to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Advanced NLP Models... Please wait.")
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

# Yahan RAM bachane ke liye distilbert use kiya hai (Yeh fast aur light hai)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')

# ==========================================
# 2. DATABASE SETUP (SIMULTANEOUS SAVING)
# ==========================================
DB_FILE = "resume_master.db"

def init_db():
    """Initialize the SQLite database and table if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            filename TEXT,
            score REAL,
            decision TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized: {DB_FILE}")

def save_candidate_to_db(name: str, filename: str, score: float, decision: str):
    """Background worker function to save data without blocking the API."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute('''
            INSERT INTO candidates (name, filename, score, decision, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, filename, score, decision, timestamp))
        conn.commit()
        print(f"Saved {name} ({decision}) to database successfully.")
    except Exception as e:
        print(f"Database Save Error: {e}")
    finally:
        conn.close()

# Run DB initialization on startup
init_db()

# ==========================================
# 3. ADVANCED PARSING ENGINE
# ==========================================
def clean_advanced_text(text: str) -> str:
    """Clean text and remove erratic whitespace."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def advanced_extract_name(text: str) -> str:
    """Multi-tier intelligent name extractor filtering out headers."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return "Unknown Candidate"
        
    blacklist = {"resume", "curriculum", "vitae", "cv", "summary", "profile", "contact", "email", "phone", "page"}
    
    # Tier 1: Look at the first 3 lines closely
    for line in lines[:3]:
        clean_line = re.sub(r'[^a-zA-Z\s]', '', line).strip()
        if clean_line.lower() in blacklist or len(clean_line.split()) < 2 or len(clean_line.split()) > 4:
            continue
            
        doc = nlp(clean_line)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
                
        # Structural fallback: Capitalized words
        words = clean_line.split()
        if all(w[0].isupper() for w in words if w.isalpha()):
            return clean_line

    # Tier 2: Search wider area using full spaCy entity verification
    full_doc = nlp(" ".join(lines[:10]))
    for ent in full_doc.ents:
        if ent.label_ == "PERSON":
            name_candidate = ent.text.strip()
            if not any(word.lower() in blacklist for word in name_candidate.split()):
                return name_candidate

    return "Unknown Candidate"

def extract_gpa(text: str) -> float:
    """Extract GPA matching strict formats."""
    pattern = r'\b(_?gpa|_?cgpa)[:\s]*([\d.]+)\b|\b([\d.]+)\s*/\s*([\d.]+)\b'
    matches = re.findall(pattern, text.lower())
    for match in matches:
        for val in match:
            if val and not val.startswith('/') and float(val) <= 10.0:
                return float(val)
    return 0.0

def get_bert_embeddings(text: str):
    """Generate PyTorch BERT embeddings."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# ==========================================
# 4. PYDANTIC MODELS FOR RESUME BUILDER
# ==========================================
class ResumeTemplateData(BaseModel):
    name: str
    email: str
    phone: str
    summary: str
    skills: list
    experience: list
    education: list
    template_id: str

# ==========================================
# 5. API ENDPOINTS
# ==========================================
@app.post("/api/screen")
async def screen_resume_endpoint(
    background_tasks: BackgroundTasks,
    job_description: str = Form(...),
    custom_keywords: str = Form(""),
    file: UploadFile = File(...)
):
    """Handles PDF uploads from Android, evaluates them, and saves to DB."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF uploads are accepted.")
        
    try:
        # Read and parse PDF
        contents = await file.read()
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)
            
        raw_text = extract_text(temp_path)
        os.remove(temp_path)
        
        if not raw_text.strip():
            raise HTTPException(status_code=420, detail="Empty or unreadable text layers inside PDF.")

        # Extract Core Features
        candidate_name = advanced_extract_name(raw_text)
        gpa_detected = extract_gpa(raw_text)
        
        # Scoring Logic
        cleaned_resume = clean_advanced_text(raw_text.lower())
        cleaned_jd = clean_advanced_text(job_description.lower())
        
        res_emb = get_bert_embeddings(cleaned_resume)
        jd_emb = get_bert_embeddings(cleaned_jd)
        similarity = float(cosine_similarity([res_emb], [jd_emb])[0][0])
        semantic_score = max(0.0, similarity) * 100
        
        # Keyword Logic
        keywords = [kw.strip().lower() for kw in custom_keywords.split(",") if kw.strip()]
        kw_hits = [kw for kw in keywords if kw in cleaned_resume]
        kw_score = (len(kw_hits) / len(keywords) * 100) if keywords else 100.0
        
        # Final AI Evaluation
        final_score = (semantic_score * 0.6) + (kw_score * 0.4)
        decision = "Selected" if final_score >= 62.0 else "Rejected"
        
        # TRIGGER BACKGROUND DB SAVE (Simultaneous execution)
        background_tasks.add_task(
            save_candidate_to_db, 
            name=candidate_name, 
            filename=file.filename, 
            score=final_score, 
            decision=decision
        )
        
        # Return instant JSON to Android
        return {
            "status": "success",
            "candidate_name": candidate_name,
            "filename": file.filename,
            "gpa": gpa_detected,
            "scores": {
                "composite": round(final_score, 1),
                "semantic": round(semantic_score, 1),
                "keyword_match": round(kw_score, 1)
            },
            "matched_keywords": kw_hits,
            "decision": decision
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-resume")
async def generate_resume_template(data: ResumeTemplateData):
    """Formats user input into a structured payload for the Android app to render."""
    enhanced_summary = f"Result-driven professional specializing in {', '.join(data.skills[:4])}. {data.summary}"
    
    return {
        "status": "success",
        "template_selected": data.template_id,
        "payload": {
            "header": {
                "name": data.name.upper(),
                "contact": f"✉ {data.email} | 📱 {data.phone}"
            },
            "summary": enhanced_summary,
            "skills_block": data.skills,
            "experience_timeline": data.experience,
            "education_details": data.education
        }
    }

if __name__ == "__main__":
    # Runs the server locally on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
