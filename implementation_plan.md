# Intelligent Resume Screening & Job Recommendation System

A Python/Streamlit NLP system that parses resumes (PDF/DOCX), extracts structured entities via spaCy NER, computes semantic match scores against job descriptions using BERT embeddings, fetches live jobs via the JSearch API, performs skill-gap analysis, and generates plain-language explanations with Gemini 2.0 Flash.

The dataset includes:
- [DataSet/Resume/Resume.csv](file:///c:/Users/FARAZ%20KHAN/Desktop/DEKSTOP/others/Major_Project/DataSet/Resume/Resume.csv) — 56MB CSV with labeled resume text across 25 job categories for model context
- `DataSet/data/data/` — ~2,400 PDF resumes across 24 categories (ACCOUNTANT, ENGINEERING, IT, etc.)
- `DataSet/data/Resumes/` — 228 real-world DOCX resumes

---

## User Review Required

> [!IMPORTANT]
> **JSearch API Key Required**: You need a RapidAPI account with JSearch API access. Insert your key in `.env`. The app will gracefully degrade to mock data if the key is absent.

> [!IMPORTANT]
> **Gemini API Key Required**: Required for the AI explanation panel. Insert in `.env`. The matching and skill-gap features still work without it.

> [!NOTE]
> **JSearch API free tier** allows 500 requests/month. The app caches results to avoid repeated identical queries.

---

## Proposed Changes

### Project Structure (NEW)

```
Major_Project/
├── app.py                    # Streamlit main app (entry point)
├── requirements.txt
├── .env.example
├── skills_kb.csv             # Skills knowledge base per job role
├── course_map.json           # Skill → Coursera/Udemy links
├── backend/
│   ├── __init__.py
│   ├── parser.py             # Text extraction from PDF/DOCX
│   ├── preprocessor.py       # NLTK cleaning pipeline
│   ├── validator.py          # Heuristic resume validation
│   ├── ner_extractor.py      # spaCy NER entity extraction
│   ├── embedder.py           # sentence-transformers embeddings
│   ├── matcher.py            # Cosine similarity match scoring
│   ├── skill_gap.py          # Skill gap analysis
│   ├── course_recommender.py # Course link recommendation
│   ├── job_fetcher.py        # JSearch API integration
│   ├── llm_explainer.py      # Gemini 2.0 Flash explanation
│   └── database.py           # SQLite persistence
└── DataSet/                  # (existing dataset)
```

---

### Backend Modules

#### [NEW] `backend/parser.py`
- Uses `pdfplumber` for PDF text extraction page-by-page
- Uses `python-docx` for DOCX extraction
- Returns raw text string from any uploaded file

#### [NEW] `backend/preprocessor.py`
- NLTK pipeline: tokenize → lowercase → remove stop-words → WordNet lemmatize
- Returns cleaned token list and joined string for downstream use

#### [NEW] `backend/validator.py`
- Heuristic scoring: checks for keywords like "experience", "skills", "education" (positive signals)
- Penalizes academic doc markers: "abstract", "conclusion", "methodology"
- Returns a score 0–100 and a `is_valid_resume` boolean

#### [NEW] `backend/ner_extractor.py`
- Uses spaCy `en_core_web_sm` model
- Regex-based email extraction (spaCy often misses emails)
- Pattern-matched phone extraction
- Keyword-list based skill extraction (mapped against a curated skills vocabulary)
- Degree extraction via pattern matching ("B.Sc", "Bachelor", "M.Tech", "PhD", etc.)
- Returns structured dict: `{name, email, phone, skills, education, organizations}`

#### [NEW] `backend/embedder.py`
- Loads `all-MiniLM-L6-v2` from `sentence-transformers`
- Exposes `encode(text)` → 384-dim numpy vector
- Singleton pattern so model loads only once

#### [NEW] `backend/matcher.py`
- `compute_similarity(resume_text, jd_text)` → cosine similarity float (0–1)
- `rank_candidates(resumes_list, jd_text)` → sorted list with percentage match scores
- Threshold ≥ 0.70 = recommended for shortlisting

#### [NEW] `backend/skill_gap.py`
- Loads `skills_kb.csv` (role → required skills mapping)
- Compares extracted candidate skills vs. role requirements
- Returns list of missing skills (gaps) per role

#### [NEW] `backend/course_recommender.py`
- Loads `course_map.json` (skill → [{platform, title, url}])
- Maps each skill gap to online learning resources
- Returns structured recommendations per missing skill

#### [NEW] `backend/job_fetcher.py`
- JSearch API via RapidAPI (queries `job_title` + `top_skills`)
- Caches last results in session state to avoid repeated API calls
- Falls back to mock jobs if API key not set

#### [NEW] `backend/llm_explainer.py`
- Uses `google-generativeai` with `gemini-2.0-flash` model
- Generates plain-language fit explanation for top match
- Example: "Matched because your Python experience aligns with ML Engineer requirements. Missing: Docker, Kubernetes."

#### [NEW] `backend/database.py`
- SQLite via Python `sqlite3` (no server needed)
- Tables: `users`, `screening_history`, `applications`
- Persists resume scans and match results for history view

---

### Data Files

#### [NEW] `skills_kb.csv`
Role-to-skills mapping (25 rows × multiple skill columns):
```
Role,Required_Skills
INFORMATION-TECHNOLOGY,"Python, SQL, Docker, Linux, Git, REST API"
ENGINEERING,"AutoCAD, MATLAB, SolidWorks, Python"
...
```

#### [NEW] `course_map.json`
```json
{
  "Docker": [{"platform": "Udemy", "title": "Docker for Beginners", "url": "https://..."}],
  "Python": [{"platform": "Coursera", "title": "Python for Everybody", "url": "https://..."}]
}
```

---

### Streamlit Application

#### [NEW] `app.py` — Main Streamlit UI

Two primary modes selectable via sidebar:

**Mode 1 — Job Seeker Mode:**
1. Upload resume PDF/DOCX → parse → validate → extract entities
2. Auto-fetch live job matches from JSearch API based on extracted job title + skills
3. Show ranked job cards with match % (cosine similarity)
4. Show Gemini AI explanation for top match
5. Show skill-gap panel + course recommendations
6. Toggle to anonymize PII (blind screening)

**Mode 2 — Recruiter Mode:**
1. Enter job description in text area
2. Upload multiple resumes (batch)
3. Rank all candidates by match score (table + bar chart)
4. Click any candidate to see detailed NER extract + skill gap
5. Export shortlist as CSV
6. PII blind screening toggle

**Shared UI Features:**
- Custom dark-mode CSS with gradient headers
- Progress bars for match scores (green ≥ 70%, amber 50–70%, red < 50%)
- Plotly charts for skill distribution and match score comparison
- Screening history sidebar (from SQLite)

---

## Verification Plan

### Automated / Script Tests

1. **Parser Test**
   ```bash
   cd c:  Desktop\DEKSTOP\others\Major_Project
   python -c "from backend.parser import extract_text; print(extract_text('DataSet/data/Resumes/AjayKumar.docx')[:500])"
   ```

2. **NER Extraction Test**
   ```bash
   python -c "
   from backend.parser import extract_text
   from backend.ner_extractor import extract_entities
   text = extract_text('DataSet/data/Resumes/AjayKumar.docx')
   print(extract_entities(text))
   "
   ```

3. **Cosine Similarity Test**
   ```bash
   python -c "
   from backend.matcher import compute_similarity
   score = compute_similarity('Python developer with Django REST API', 'Looking for Python Django backend developer')
   print(f'Match Score: {score*100:.1f}%')
   "
   ```

4. **Validator Test**
   ```bash
   python -c "
   from backend.validator import validate_resume
   print(validate_resume('This paper presents an abstract and conclusion of research on NLP'))
   print(validate_resume('5 years experience as Python developer. Skills: Python, SQL, Django'))
   "
   ```

### Manual Verification (Browser / Streamlit)

1. Run the app:
   ```bash
   cd c:\Users\FARAZ KHAN\Desktop\DEKSTOP\others\Major_Project
   streamlit run app.py
   ```
2. Open browser at `http://localhost:8501`
3. **Test Job Seeker Mode**: Upload any `.docx` from `DataSet/data/Resumes/` → verify entity extraction panel shows Name, Skills, Email
4. **Test Recruiter Mode**: Paste a sample JD → upload 3–5 resumes → verify ranking table appears with match %
5. **Test Skill Gap Panel**: Confirm missing skills and course links appear for the top candidate
6. **Test Blind Screening**: Toggle PII anonymization → verify name/email fields are masked
