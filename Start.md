Backend NLP Improvements (6 modules)


Module	What Changed
category_classifier.py (NEW)	Random Forest + TF-IDF trained on your Resume.csv dataset — predicts which of 25 job categories a resume belongs to
train_classifier.py (NEW)	One-time training script, saves models to models/ directory
ats_scorer.py (NEW)	7-dimension ATS compatibility scoring (Contact, Skills, Experience, Education, Format, Keywords, Links) → Grade A–F
vector_store.py (NEW)	ChromaDB vector store for persistent resume search (with in-memory fallback)
ner_extractor.py (ENHANCED)	200+ skills vocab, experience year extraction, job title detection, LinkedIn/GitHub links, frequency-based skill proficiency (Expert → Familiar)
matcher.py (ENHANCED)	Hybrid scoring: 60% BERT semantic + 25% skill overlap + 15% category alignment
UI Improvements
Tab-based layout: Profile & ATS → Match Analysis → Upskill Path → Live Jobs
Category badge with confidence bar and top-3 predictions
ATS Score Ring with grade and 7-dimension breakdown bars
Skill proficiency dots (🟢 Expert, 🔵 Proficient, 🟡 Intermediate, ⚪ Familiar)
Pipeline visualization showing the NLP processing steps
Hybrid score breakdown cards showing semantic/skill/category component scores

Polished dark mode with deeper gradients and refined glassmorphism
To run it:


bash
python -m streamlit run app.py



