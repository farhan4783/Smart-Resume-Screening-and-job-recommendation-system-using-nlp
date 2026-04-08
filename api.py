"""
api.py
------
FastAPI REST API for the Intelligent Resume Screening System.
Wraps all existing backend modules and serves the React frontend.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
import io
import json
import hashlib
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from backend.parser import extract_text
from backend.preprocessor import preprocess
from backend.validator import validate_resume
from backend.ner_extractor import extract_entities
from backend.matcher import compute_similarity, compute_hybrid_score, rank_candidates, get_match_label
from backend.skill_gap import analyze_gap, infer_role_from_skills, get_all_roles
from backend.course_recommender import recommend_courses
from backend.job_fetcher import fetch_jobs
from backend.llm_explainer import explain_match, generate_resume_feedback
from backend.category_classifier import predict_category
from backend.ats_scorer import compute_ats_score
from backend import database as db

# ─── App Init ────────────────────────────────────────────────────────────────

app = FastAPI(title="ResumeIQ API", version="2.0")

# CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _process_file(file: UploadFile, anonymize: bool = False) -> dict:
    """Process an uploaded resume file through the full NLP pipeline."""
    contents = file.file.read()
    file_like = io.BytesIO(contents)
    file_like.name = file.filename

    raw_text = extract_text(file_like, file_name=file.filename)
    if not raw_text or len(raw_text.strip()) < 20:
        raise HTTPException(status_code=400, detail=f"Could not extract text from {file.filename}")

    validation = validate_resume(raw_text)
    entities = extract_entities(raw_text, anonymize_pii=anonymize)
    _, clean = preprocess(raw_text)
    category = predict_category(raw_text)

    links_list = []
    if entities.get("links"):
        if entities["links"].get("linkedin"):
            links_list.append(entities["links"]["linkedin"])
        if entities["links"].get("github"):
            links_list.append(entities["links"]["github"])

    ats = compute_ats_score(
        text=raw_text,
        skills=entities["skills"],
        education=entities["education"],
        experience_years=entities["experience_years"],
        email=entities["email"],
        phone=entities["phone"],
        name=entities["name"],
        links=links_list,
    )

    return {
        "filename": file.filename,
        "raw_text": raw_text,
        "clean_text": clean,
        "validation": validation,
        "entities": entities,
        "category": category,
        "ats": ats,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0"}


@app.post("/api/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    anonymize: bool = Form(False),
):
    """
    Upload a resume → full NLP pipeline.
    Returns: entities, category, ATS score, validation.
    """
    result = _process_file(file, anonymize)
    # Don't send raw_text back (too large), keep a hash for reference
    return {
        "filename": result["filename"],
        "validation": result["validation"],
        "entities": result["entities"],
        "category": result["category"],
        "ats": result["ats"],
        "text_length": len(result["raw_text"]),
        "text_hash": hashlib.md5(result["raw_text"].encode()).hexdigest(),
    }


@app.post("/api/match")
async def match_resume(
    file: UploadFile = File(...),
    job_description: str = Form(""),
    target_role: str = Form(""),
    location: str = Form("India"),
    anonymize: bool = Form(False),
):
    """
    Upload resume + JD → hybrid match, skill gap, job search, AI explanation.
    """
    result = _process_file(file, anonymize)
    entities = result["entities"]
    raw_text = result["raw_text"]
    category = result["category"]

    # Determine role
    if not target_role:
        target_role = infer_role_from_skills(entities["skills"])

    # Skill gap
    gap = analyze_gap(entities["skills"], target_role)

    # Job search
    job_title_query = target_role.replace("-", " ").title()
    top_skills = entities["skills"][:5]
    jobs = fetch_jobs(job_title_query, top_skills, location=location)

    # Score jobs
    scored_jobs = []
    for job in jobs[:8]:
        jd = job.get("job_description", job.get("job_title", ""))
        if jd:
            jd_to_use = job_description if job_description else jd
            hybrid = compute_hybrid_score(
                resume_text=raw_text,
                jd_text=jd_to_use,
                candidate_skills=entities["skills"],
                required_skills=gap.get("matched_skills", []) + gap.get("missing_skills", []),
                category_match=(category["category"].upper() == target_role.upper()),
            )
            label, color = get_match_label(hybrid["total_score"])
            scored_jobs.append({
                "job_title": job.get("job_title", "N/A"),
                "employer_name": job.get("employer_name", "N/A"),
                "job_city": job.get("job_city", "N/A"),
                "job_employment_type": job.get("job_employment_type", ""),
                "job_posted_at": job.get("job_posted_at_datetime_utc", job.get("job_posted_at", "")),
                "job_apply_link": job.get("job_apply_link", "#"),
                "job_description": jd[:500],
                "score": hybrid["total_score"],
                "semantic_score": hybrid["semantic_score"],
                "skill_score": hybrid["skill_score"],
                "category_score": hybrid["category_score"],
                "label": label,
                "color": color,
            })
        else:
            scored_jobs.append({
                "job_title": job.get("job_title", "N/A"),
                "employer_name": job.get("employer_name", "N/A"),
                "score": 0.0,
                "label": "N/A",
                "color": "#64748b",
                "job_apply_link": job.get("job_apply_link", "#"),
            })

    scored_jobs.sort(key=lambda x: x.get("score", 0), reverse=True)

    # AI explanation for top job
    explanation = ""
    if scored_jobs and scored_jobs[0].get("score", 0) > 0:
        try:
            explanation = explain_match(
                candidate_name=entities["name"],
                resume_text=raw_text,
                job_description=scored_jobs[0].get("job_description", ""),
                match_score=scored_jobs[0]["score"],
                candidate_skills=entities["skills"],
                job_title=scored_jobs[0].get("job_title", job_title_query),
                missing_skills=gap["missing_skills"],
            )
        except Exception as e:
            explanation = f"Could not generate explanation: {e}"

    # AI resume feedback
    resume_feedback = ""
    try:
        resume_feedback = generate_resume_feedback(
            raw_text,
            scored_jobs[0].get("job_description", "") if scored_jobs else ""
        )
    except Exception:
        pass

    # Course recommendations
    course_recs = []
    if gap["missing_skills"]:
        course_recs = recommend_courses(gap["missing_skills"][:6])

    # Save to DB
    db.save_screening(
        filename=result["filename"],
        candidate=entities["name"],
        skills=entities["skills"],
        match_score=scored_jobs[0]["score"] if scored_jobs else 0,
        job_title=job_title_query,
        mode="seeker",
        is_valid=result["validation"]["is_valid"],
    )

    return {
        "filename": result["filename"],
        "validation": result["validation"],
        "entities": entities,
        "category": category,
        "ats": result["ats"],
        "gap": gap,
        "jobs": scored_jobs,
        "explanation": explanation,
        "resume_feedback": resume_feedback,
        "courses": course_recs,
        "target_role": target_role,
    }


@app.post("/api/rank")
async def rank_resumes(
    files: List[UploadFile] = File(...),
    job_description: str = Form(...),
    target_role: str = Form(""),
    anonymize: bool = Form(False),
):
    """
    Upload multiple resumes + JD → ranked candidates.
    """
    if not job_description:
        raise HTTPException(status_code=400, detail="Job description is required")

    candidates = []
    for f in files:
        try:
            result = _process_file(f, anonymize)
            candidates.append({
                "name": result["entities"]["name"] or f.filename,
                "filename": f.filename,
                "text": result["raw_text"],
                "skills": result["entities"]["skills"],
                "email": result["entities"]["email"],
                "education": result["entities"]["education"],
                "experience_years": result["entities"]["experience_years"],
                "validation": result["validation"],
                "predicted_category": result["category"]["category"],
                "category_confidence": result["category"]["confidence"],
                "ats_score": result["ats"]["total_score"],
                "ats_grade": result["ats"]["grade"],
                "job_titles": result["entities"].get("job_titles", []),
            })
        except Exception as e:
            candidates.append({
                "name": f.filename,
                "filename": f.filename,
                "error": str(e),
            })

    # Get required skills from knowledge base
    if not target_role:
        target_role = get_all_roles()[0] if get_all_roles() else ""

    temp_gap = analyze_gap([], target_role)
    req_skills = temp_gap.get("missing_skills", [])

    valid_candidates = [c for c in candidates if "text" in c]
    ranked = rank_candidates(valid_candidates, job_description, required_skills=req_skills)

    # Enrich with skill gaps and AI explanations
    enriched = []
    for r in ranked:
        gap = analyze_gap(r.get("skills", []), target_role)
        label, color = get_match_label(r["match_score"])

        # AI explanation
        ai_explanation = ""
        try:
            ai_explanation = explain_match(
                candidate_name=r["name"],
                resume_text=r["text"],
                job_description=job_description,
                match_score=r["match_score"],
                candidate_skills=r.get("skills", []),
                job_title=target_role.replace("-", " ").title(),
                missing_skills=gap["missing_skills"],
            )
        except Exception:
            pass

        enriched.append({
            "name": r["name"],
            "filename": r["filename"],
            "match_score": r["match_score"],
            "semantic_score": r.get("semantic_score", 0),
            "skill_overlap": r.get("skill_overlap", 0),
            "category_bonus": r.get("category_bonus", False),
            "shortlisted": r.get("shortlisted", False),
            "label": label,
            "color": color,
            "skills": r.get("skills", []),
            "email": r.get("email", ""),
            "education": r.get("education", []),
            "experience_years": r.get("experience_years", 0),
            "predicted_category": r.get("predicted_category", ""),
            "ats_score": r.get("ats_score", 0),
            "ats_grade": r.get("ats_grade", ""),
            "validation": r.get("validation", {}),
            "gap": gap,
            "ai_explanation": ai_explanation,
        })

        # Save to DB
        db.save_screening(
            filename=r["filename"],
            candidate=r["name"],
            skills=r.get("skills", []),
            match_score=r["match_score"],
            job_title=target_role,
            mode="recruiter",
            is_valid=r.get("validation", {}).get("is_valid", False),
        )

    return {
        "candidates": enriched,
        "total": len(enriched),
        "shortlisted": sum(1 for c in enriched if c["shortlisted"]),
        "avg_score": sum(c["match_score"] for c in enriched) / len(enriched) if enriched else 0,
        "top_score": enriched[0]["match_score"] if enriched else 0,
        "target_role": target_role,
    }


@app.get("/api/roles")
def get_roles():
    """Get all role categories from knowledge base."""
    return {"roles": get_all_roles()}


@app.get("/api/history")
def get_history(limit: int = 20):
    """Get screening history."""
    return {"history": db.get_history(limit=limit)}


@app.delete("/api/history")
def clear_history():
    """Clear all screening history."""
    db.clear_history()
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
