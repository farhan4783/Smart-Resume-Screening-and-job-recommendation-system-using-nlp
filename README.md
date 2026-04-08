# 🧠 ResumeIQ: Intelligent Resume Screening & Job Recommendation System

![ResumeIQ Architecture](https://img.shields.io/badge/Architecture-React%20%2B%20FastAPI-blue?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-spaCy%20%7C%20BERT-green?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange?style=for-the-badge)
![AI](https://img.shields.io/badge/Generative%20AI-Gemini%202.0-purple?style=for-the-badge)

**ResumeIQ** is an advanced, AI-driven recruitment ecosystem designed to transition hiring pipelines from legacy keyword-filtering to **deep semantic understanding**. By treating resumes as dynamic semantic profiles rather than static text documents, it completely eliminates the "keyword rigidity" problem found in traditional Applicant Tracking Systems (ATS).

This project was built for the Major Project Session 2025-26 under the School of Computer Science and Engineering, IILM University.

---

## ✨ Key Features

### 🎯 Job Seeker Dashboard
- **Profile & ATS Extraction:** Automatically extracts an applicant's core competencies (from over 200+ normalized tech skills), evaluates proficiency (Expert, Proficient, Intermediate, Familiar) based on occurrence frequency, and calculates a comprehensive ATS Compatibility Score across 7 dimensions.
- **Category Prediction:** Employs a trained Random Forest + TF-IDF classifier to predict the applicant's role category (out of 25 roles) along with an AI confidence score.
- **Match Analysis & Gap Detection:** Calculates a Hybrid Score between your resume and a target job description, identifying exact matched skills and pinpointing critical missing skills.
- **AI Match Explanation:** Integrates Google's Gemini LLM to generate a natural-language explanation of *why* your resume is a strong (or weak) match, alongside actionable CV feedback.
- **Upskill Path Recommendations:** Dynamically recommends relevant learning paths (e.g., Coursera) based precisely on the skill gaps detected during your match analysis.
- **Live Job Hunting:** Connects to the RapidAPI JSearch API to fetch real-world, live job postings aligned with your skills and target location, pre-scored using the hybrid matching engine.

### 🏢 Recruiter Dashboard
- **Multi-Candidate Processing:** Upload multiple candidate resumes (PDF/DOCX) simultaneously and input a target Job Description.
- **Hybrid Candidate Ranking:** Automatically scores and ranks all candidates using our custom matching algorithm:
  - **60% Semantic Similarity:** (BERT embeddings via `all-MiniLM-L6-v2`)
  - **25% Skill Overlap:** (Jaccard-like flexible substring overlap)
  - **15% Category Alignment:** (Machine Learning prediction alignment)
- **Deep Dives & Visualizations:** Interactive data tables, interactive bar charts for score comparison, and expandable candidate deep-dives containing ATS breakdowns, gap analyses, and AI-generated rationales.
- **Blind Screening Mode:** One-click toggle to mask Personally Identifiable Information (PII) like names, emails, and phone numbers to mitigate implicit biological or demographic bias during early funnel screening.

---

## 🏗️ System Architecture

The application has been modularized into a cloud-native architecture consisting of a Python NLP API Backend and a modern React Vite Frontend.

```
ResumeIQ/
├── backend/              # Core NLP & ML Modules
│   ├── ner_extractor.py  # Named Entity Recognition (spaCy)
│   ├── embedder.py       # Sentence Transformers (BERT)
│   ├── matcher.py        # Hybrid Match Algorithm
│   ├── ats_scorer.py     # 7-Dimension Rubric Scorer
│   ├── category_classifier.py # ML Prediction
│   └── ...
├── api.py                # FastAPI REST endpoints
├── frontend/             # React SPA (Vite)
│   ├── src/App.jsx       # Complete Dashboard UI
│   ├── src/index.css     # Dark Glassmorphism Design System
│   └── src/api.js        # API Client wrapper
├── models/               # Saved RF/TF-IDF models (.pkl)
├── requirements.txt      # Python dependencies
└── package.json          # Node dependencies (in frontend/)
```

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.9+** (For the NLP backend)
- **Node.js 18+** (For the React frontend)
- API Keys: [Google Gemini](https://aistudio.google.com/app/apikey) & [RapidAPI JSearch](https://rapidapi.com/letscrape-6bRBa3QGzB0/api/jsearch)

### 1. Backend Setup (FastAPI & NLP)

```bash
# Clone the repository
git clone https://github.com/yourusername/ResumeIQ.git
cd ResumeIQ

# Install required Python dependencies
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart

# Download the spaCy English model
python -m spacy download en_core_web_sm

# Configure Environment Variables
# Copy the example env file and insert your API keys
cp .env.example .env

# Train the Category Classifier (Only required once)
# Ensure the Resume.csv dataset is present in DataSet/data/
python -m backend.train_classifier

# Start the FastAPI Server (Starts on port 8000)
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup (React & Vite)

```bash
# Open a new terminal session
cd ResumeIQ/frontend

# Install core Node dependencies
npm install

# Start the Vite Development Server (Starts on port 5173)
npm run dev
```

Finally, open your browser and navigate to `http://localhost:5173`. 

---

## 🛠️ Tech Stack & Technologies

### Backend (Python)
- **Framework:** FastAPI
- **Machine Learning:** Scikit-Learn (Random Forest, TF-IDF)
- **NLP & Embeddings:** spaCy, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Document Parsing:** pdfplumber, PyMuPDF, python-docx
- **Generative AI:** Google Generative AI (Gemini Flash)
- **Database:** SQLite3 / ChromaDB

### Frontend (JavaScript)
- **Framework:** React 18 (via Vite)
- **Design System:** Custom Dark Glassmorphism (Vanilla CSS variables)
- **HTTP Client:** Fetch API (Proxy configured to avoid CORS during dev)

---

## 👥 Project Team

**Project Guide:** Ms. Aanchal Vij  
**Institution:** School of Computer Science and Engineering, IILM University

**Team Members:**
- Mohd Farhan (CS-2341429)
- Navaneet (CS-23411348)
- Navya Vashistha (CS-23411266)
- Mohammad Umar (CS-2341437)
- Nida Fatima (CS-2341757)
