"""
app.py
------
Intelligent Resume Screening & Job Recommendation System
Main Streamlit Application
"""

# Suppress TensorFlow before any ML library imports (prevents DLL errors on Windows)
import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
import io
import json
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

# Ensure backend module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from backend.parser import extract_text
from backend.preprocessor import preprocess
from backend.validator import validate_resume
from backend.ner_extractor import extract_entities
from backend.matcher import compute_similarity, rank_candidates, get_match_label
from backend.skill_gap import analyze_gap, infer_role_from_skills, get_all_roles
from backend.course_recommender import recommend_courses
from backend.job_fetcher import fetch_jobs
from backend.llm_explainer import explain_match, generate_resume_feedback
from backend import database as db

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ — Intelligent Resume Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Initialize DB ────────────────────────────────────────────────────────────
db.init_db()

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Match score badges */
    .badge-green  { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid #22c55e; border-radius: 8px; padding: 4px 12px; font-weight: 600; }
    .badge-amber  { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid #f59e0b; border-radius: 8px; padding: 4px 12px; font-weight: 600; }
    .badge-red    { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid #ef4444; border-radius: 8px; padding: 4px 12px; font-weight: 600; }

    /* Skill tag */
    .skill-tag {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(118,75,162,0.25));
        color: #c4b5fd;
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        margin: 3px;
        font-weight: 500;
    }

    /* Missing skill tag */
    .skill-missing {
        display: inline-block;
        background: rgba(239,68,68,0.1);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        margin: 3px;
        font-weight: 500;
    }

    /* Section header */
    .section-title {
        color: #e2e8f0;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1rem 0 0.5rem;
        border-left: 4px solid #6366f1;
        padding-left: 0.75rem;
    }

    /* Job card */
    .job-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(118,75,162,0.08));
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px;
        padding: 1.25rem;
        margin: 0.6rem 0;
        transition: all 0.2s;
    }
    .job-card:hover {
        border-color: rgba(99,102,241,0.5);
        box-shadow: 0 4px 20px rgba(99,102,241,0.15);
    }

    /* Override Streamlit defaults */
    .stTextArea textarea, .stTextInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #e2e8f0 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
    }
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
    }
    h1, h2, h3 { color: #e2e8f0 !important; }
    p, li, span { color: #cbd5e1; }
    .stMarkdown p { color: #cbd5e1; }
    hr { border-color: rgba(255,255,255,0.1) !important; }
    .uploadedFile { background: rgba(255,255,255,0.05) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def render_skill_tags(skills: list, missing=False):
    cls = "skill-missing" if missing else "skill-tag"
    tags = " ".join(f'<span class="{cls}">{s}</span>' for s in skills)
    st.markdown(tags, unsafe_allow_html=True)


def score_color(score: float) -> str:
    if score >= 0.70: return "#22c55e"
    if score >= 0.50: return "#f59e0b"
    return "#ef4444"


def make_gauge(score: float, title: str = "Match Score") -> go.Figure:
    pct = score * 100
    color = score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        delta={"reference": 70, "valueformat": ".1f", "suffix": "%"},
        title={"text": title, "font": {"color": "#94a3b8", "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],  "color": "rgba(239,68,68,0.1)"},
                {"range": [50, 70], "color": "rgba(245,158,11,0.1)"},
                {"range": [70, 100],"color": "rgba(34,197,94,0.1)"},
            ],
            "threshold": {
                "line": {"color": "#6366f1", "width": 3},
                "thickness": 0.8, "value": 70
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=20, r=20),
        height=250,
        font_color="#e2e8f0",
    )
    return fig


def make_skills_bar(matched: list, missing: list) -> go.Figure:
    labels = ["Matched Skills", "Missing Skills"]
    values = [len(matched), len(missing)]
    colors = ["#22c55e", "#ef4444"]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=values, textposition="auto",
        textfont=dict(color="white", size=14),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        margin=dict(t=20, b=10, l=10, r=10),
        height=220,
        showlegend=False,
        xaxis=dict(showgrid=False, color="#94a3b8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#94a3b8"),
    )
    return fig


def process_resume(uploaded_file, anonymize: bool = False) -> dict:
    """Full pipeline: parse → validate → NER → return results dict."""
    raw_text = extract_text(uploaded_file, file_name=uploaded_file.name)
    validation = validate_resume(raw_text)
    entities = extract_entities(raw_text, anonymize_pii=anonymize)
    _, clean = preprocess(raw_text)
    return {
        "filename": uploaded_file.name,
        "raw_text": raw_text,
        "clean_text": clean,
        "validation": validation,
        "entities": entities,
    }


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2.5rem;">🧠</div>
        <div style="font-size:1.1rem; font-weight:700; color:#c4b5fd;">ResumeIQ</div>
        <div style="font-size:0.75rem; color:#64748b;">Intelligent Resume Screening</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    mode = st.radio(
        "**Select Mode**",
        ["🎯 Job Seeker", "🏢 Recruiter"],
        help="Job Seeker: upload your resume & find matching jobs. Recruiter: rank multiple candidates against a JD."
    )

    st.markdown("---")
    st.markdown("**⚙️ Options**")
    anonymize_pii = st.toggle(
        "🙈 Blind Screening (Hide PII)",
        value=False,
        help="Anonymizes Name, Email, Phone to ensure bias-free screening based on skills only."
    )

    st.markdown("---")
    st.markdown("**📋 Screening History**")
    history = db.get_history(limit=8)
    if history:
        for h in history[:5]:
            score_str = f"{h['match_score']*100:.0f}%" if h.get("match_score") else "N/A"
            st.markdown(
                f"<small style='color:#64748b;'>📄 {h['filename'][:20]}… &nbsp;|&nbsp; {score_str}</small>",
                unsafe_allow_html=True
            )
        if st.button("🗑️ Clear History", key="clear_hist"):
            db.clear_history()
            st.rerun()
    else:
        st.markdown("<small style='color:#475569;'>No history yet</small>", unsafe_allow_html=True)


# ─── HERO HEADER ─────────────────────────────────────────────────────────────

st.markdown("""
<div class='hero-header'>Intelligent Resume Screening<br>& Job Recommendation</div>
<div class='hero-sub'>🚀 Powered by BERT Semantic Embeddings · spaCy NER · Gemini AI · JSearch Live Jobs</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# MODE 1: JOB SEEKER
# ════════════════════════════════════════════════════════════════════════════

if mode == "🎯 Job Seeker":
    st.markdown("## 🎯 Job Seeker Dashboard")
    st.markdown("Upload your resume to get live job matches, AI-powered insights, and skill-gap analysis.")

    uploaded = st.file_uploader(
        "📄 Upload Resume (PDF or DOCX)",
        type=["pdf", "docx", "doc"],
        key="seeker_upload",
        help="Your resume data is processed locally and never stored in the cloud."
    )

    if uploaded:
        with st.spinner("🔍 Parsing and analyzing your resume..."):
            result = process_resume(uploaded, anonymize=anonymize_pii)
            entities = result["entities"]
            validation = result["validation"]
            raw_text = result["raw_text"]

        # ── Validation Banner ──────────────────────────────────────────────
        if validation["is_valid"]:
            st.success(f"✅ Valid Resume Detected — {validation['reason']}")
        else:
            st.warning(f"⚠️ {validation['reason']} — Results may be less accurate.")

        # ── Entity Summary Row ─────────────────────────────────────────────
        st.markdown("<div class='section-title'>📊 Extracted Profile</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👤 Name", entities["name"] or "Not Detected")
        c2.metric("📧 Email", entities["email"] or "Not Found")
        c3.metric("📱 Phone", entities["phone"] or "Not Found")
        c4.metric("🎓 Education", ", ".join(entities["education"][:2]) or "Not Detected")

        # ── Skills ────────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>🛠️ Detected Skills</div>", unsafe_allow_html=True)
        if entities["skills"]:
            render_skill_tags(entities["skills"])
        else:
            st.info("No specific skills detected. Try a more detailed resume.")

        # ── Infer role & skill gap ─────────────────────────────────────────
        all_roles = get_all_roles()
        inferred_role = infer_role_from_skills(entities["skills"])

        st.markdown("---")
        col_role, col_loc = st.columns([2, 1])
        with col_role:
            selected_role = st.selectbox(
                "🎯 Target Role for Skill Gap Analysis",
                options=all_roles,
                index=all_roles.index(inferred_role) if inferred_role in all_roles else 0,
                help="Auto-inferred from your resume. Change to analyze gaps for a different role."
            )
        with col_loc:
            location = st.text_input("📍 Job Location", value="India")

        if st.button("🚀 Analyze My Resume & Find Jobs", key="seeker_analyze", use_container_width=True):
            # ── Skill Gap ──────────────────────────────────────────────────
            with st.spinner("🔬 Running skill gap analysis..."):
                gap = analyze_gap(entities["skills"], selected_role)

            st.markdown("<div class='section-title'>📈 Skill Coverage Analysis</div>", unsafe_allow_html=True)
            cg1, cg2, cg3 = st.columns(3)
            cg1.metric("✅ Skills Matched", len(gap["matched_skills"]))
            cg2.metric("❌ Skills Missing", len(gap["missing_skills"]))
            cg3.metric("📊 Coverage", f"{gap['coverage_pct']}%")

            col_chart, col_gap = st.columns([1, 1])
            with col_chart:
                st.plotly_chart(make_skills_bar(gap["matched_skills"], gap["missing_skills"]), use_container_width=True)
            with col_gap:
                if gap["matched_skills"]:
                    st.markdown("**✅ You Have:**")
                    render_skill_tags(gap["matched_skills"])
                if gap["missing_skills"]:
                    st.markdown("**❌ You're Missing:**")
                    render_skill_tags(gap["missing_skills"], missing=True)

            # ── Course Recommendations ─────────────────────────────────────
            if gap["missing_skills"]:
                st.markdown("<div class='section-title'>📚 Recommended Courses to Close Gaps</div>", unsafe_allow_html=True)
                courses = recommend_courses(gap["missing_skills"][:6])
                for item in courses:
                    with st.expander(f"📖 {item['skill']}", expanded=False):
                        for course in item["courses"]:
                            st.markdown(
                                f"🎓 **{course['platform']}** — [{course['title']}]({course['url']})"
                            )

            # ── Live Job Fetch ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("<div class='section-title'>💼 Live Job Recommendations</div>", unsafe_allow_html=True)

            job_title_query = selected_role.replace("-", " ").title()
            top_skills = entities["skills"][:5]

            with st.spinner(f"🔎 Fetching live {job_title_query} jobs..."):
                jobs = fetch_jobs(job_title_query, top_skills, location=location)

            if not jobs:
                st.info("No jobs found. Try changing the role or location.")
            else:
                # Compute match scores for each job
                job_texts = []
                for job in jobs[:8]:
                    jd = job.get("job_description", job.get("job_title", ""))
                    job_texts.append(jd)

                with st.spinner("🧮 Computing semantic match scores..."):
                    scored_jobs = []
                    for job, jd_text in zip(jobs[:8], job_texts):
                        score = compute_similarity(raw_text, jd_text) if jd_text else 0.0
                        scored_jobs.append({**job, "_score": score})
                    scored_jobs.sort(key=lambda x: x["_score"], reverse=True)

                top_job = scored_jobs[0] if scored_jobs else None

                col_gauge, col_jobs = st.columns([1, 2])
                with col_gauge:
                    if top_job:
                        st.plotly_chart(make_gauge(top_job["_score"], "Top Job Match"), use_container_width=True)
                        label, _ = get_match_label(top_job["_score"])
                        st.markdown(f"<center style='color:#94a3b8;'>{label}</center>", unsafe_allow_html=True)

                with col_jobs:
                    for job in scored_jobs[:5]:
                        score = job["_score"]
                        label, color = get_match_label(score)
                        city   = job.get("job_city", "N/A")
                        etype  = job.get("job_employment_type", "").replace("_", " ").title()
                        posted = job.get("job_posted_at_datetime_utc", job.get("job_posted_at", ""))
                        if isinstance(posted, str) and "T" in posted:
                            posted = posted.split("T")[0]

                        st.markdown(f"""
                        <div class='job-card'>
                            <div style='display:flex; justify-content:space-between; align-items:center;'>
                                <div>
                                    <div style='font-weight:700; color:#e2e8f0; font-size:1rem;'>{job.get('job_title','N/A')}</div>
                                    <div style='color:#94a3b8; font-size:0.85rem;'>🏢 {job.get('employer_name','N/A')} &nbsp;|&nbsp; 📍 {city} &nbsp;|&nbsp; {etype}</div>
                                    <div style='color:#64748b; font-size:0.78rem; margin-top:4px;'>📅 {posted}</div>
                                </div>
                                <div style='text-align:right;'>
                                    <div style='font-size:1.4rem; font-weight:800; color:{color};'>{score*100:.0f}%</div>
                                    <div style='font-size:0.72rem; color:{color};'>{label}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        apply_link = job.get("job_apply_link", "#")
                        st.markdown(f"[🔗 Apply Now]({apply_link})", unsafe_allow_html=True)

                # ── AI Explanation (top match) ─────────────────────────────
                if top_job:
                    st.markdown("---")
                    st.markdown("<div class='section-title'>🤖 AI Match Explanation (Top Job)</div>", unsafe_allow_html=True)
                    with st.spinner("✨ Generating AI explanation via Gemini..."):
                        explanation = explain_match(
                            candidate_name=entities["name"],
                            resume_text=raw_text,
                            job_description=top_job.get("job_description", ""),
                            match_score=top_job["_score"],
                            candidate_skills=entities["skills"],
                            job_title=top_job.get("job_title", job_title_query),
                            missing_skills=gap["missing_skills"],
                        )
                    st.markdown(f"""
                    <div class='glass-card'>
                        {explanation.replace(chr(10), '<br>')}
                    </div>""", unsafe_allow_html=True)

                # ── Resume Feedback ────────────────────────────────────────
                st.markdown("---")
                st.markdown("<div class='section-title'>💡 AI Resume Improvement Tips</div>", unsafe_allow_html=True)
                with st.spinner("📝 Generating resume feedback..."):
                    feedback = generate_resume_feedback(raw_text, top_job.get("job_description", "") if top_job else "")
                st.markdown(f"""
                <div class='glass-card'>
                    {feedback.replace(chr(10), '<br>')}
                </div>""", unsafe_allow_html=True)

                # ── Save to DB ─────────────────────────────────────────────
                db.save_screening(
                    filename=result["filename"],
                    candidate=entities["name"],
                    skills=entities["skills"],
                    match_score=top_job["_score"] if top_job else 0,
                    job_title=job_title_query,
                    mode="seeker",
                    is_valid=validation["is_valid"],
                )

    else:
        # Placeholder state
        st.markdown("""
        <div style='text-align:center; padding: 4rem 2rem; color: #475569;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>📄</div>
            <div style='font-size: 1.2rem; font-weight: 600; color: #64748b;'>Upload your resume to get started</div>
            <div style='margin-top: 0.5rem; font-size: 0.9rem;'>Supports PDF and DOCX formats</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODE 2: RECRUITER
# ════════════════════════════════════════════════════════════════════════════

elif mode == "🏢 Recruiter":
    st.markdown("## 🏢 Recruiter Dashboard")
    st.markdown("Enter a job description and upload candidate resumes to rank them by semantic fit.")

    col_jd, col_upload = st.columns([1, 1])

    with col_jd:
        st.markdown("**📋 Job Description**")
        jd_text = st.text_area(
            label="Paste the job description here",
            placeholder="e.g. We are looking for a Senior Python Developer with 5+ years of experience in Django, REST APIs, Docker, and AWS...",
            height=300,
            key="recruiter_jd",
            label_visibility="collapsed",
        )

    with col_upload:
        st.markdown("**📂 Candidate Resumes (multiple)**")
        uploaded_files = st.file_uploader(
            "Upload resumes",
            type=["pdf", "docx", "doc"],
            accept_multiple_files=True,
            key="recruiter_upload",
            label_visibility="collapsed",
        )

    target_role = st.selectbox(
        "🎯 Role Category (for skill-gap analysis)",
        options=get_all_roles(),
        index=0,
        key="recruiter_role",
    )

    run_btn = st.button(
        "🏃 Rank Candidates",
        key="recruiter_run",
        disabled=(not jd_text or not uploaded_files),
        use_container_width=True,
    )

    if run_btn and jd_text and uploaded_files:
        candidates = []

        progress_bar = st.progress(0, text="Parsing resumes...")
        total = len(uploaded_files)

        for i, f in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / total, text=f"Parsing: {f.name}")
            try:
                res = process_resume(f, anonymize=anonymize_pii)
                candidates.append({
                    "name": res["entities"]["name"] or f.name,
                    "filename": f.name,
                    "text": res["raw_text"],
                    "skills": res["entities"]["skills"],
                    "email": res["entities"]["email"],
                    "education": res["entities"]["education"],
                    "validation": res["validation"],
                })
            except Exception as e:
                st.warning(f"Could not parse {f.name}: {e}")

        progress_bar.progress(1.0, text="Computing match scores...")

        with st.spinner("🧮 Computing semantic similarity scores..."):
            ranked = rank_candidates(candidates, jd_text)

        progress_bar.empty()

        st.markdown("---")
        st.markdown("<div class='section-title'>🏆 Candidate Ranking</div>", unsafe_allow_html=True)

        # ── Summary metrics ────────────────────────────────────────────────
        shortlisted = [r for r in ranked if r.get("shortlisted")]
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("📋 Total Candidates", len(ranked))
        sm2.metric("✅ Shortlisted (≥70%)", len(shortlisted))
        sm3.metric("🥇 Top Score", f"{ranked[0]['match_score']*100:.1f}%" if ranked else "N/A")
        sm4.metric("📊 Avg Score", f"{sum(r['match_score'] for r in ranked)/len(ranked)*100:.1f}%" if ranked else "N/A")

        # ── Ranking table ──────────────────────────────────────────────────
        table_data = []
        for rank_i, r in enumerate(ranked, 1):
            label, _ = get_match_label(r["match_score"])
            table_data.append({
                "Rank": rank_i,
                "Candidate": r["name"],
                "File": r["filename"],
                "Match Score": f"{r['match_score']*100:.1f}%",
                "Status": label,
                "Skills Found": len(r.get("skills", [])),
                "Shortlisted": "✅ Yes" if r.get("shortlisted") else "❌ No",
            })

        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Match Score": st.column_config.TextColumn("Match Score"),
                "Status": st.column_config.TextColumn("Status"),
            },
        )

        # ── Export CSV ─────────────────────────────────────────────────────
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Shortlist CSV",
            data=csv_data,
            file_name="candidate_ranking.csv",
            mime="text/csv",
        )

        # ── Score comparison bar chart ─────────────────────────────────────
        st.markdown("<div class='section-title'>📊 Score Comparison</div>", unsafe_allow_html=True)
        names_chart = [r["name"][:20] for r in ranked]
        scores_chart = [r["match_score"] * 100 for r in ranked]
        colors_chart = [score_color(r["match_score"]) for r in ranked]

        fig_bar = go.Figure(go.Bar(
            x=names_chart, y=scores_chart,
            marker_color=colors_chart,
            text=[f"{s:.1f}%" for s in scores_chart],
            textposition="auto",
            textfont=dict(color="white"),
        ))
        fig_bar.add_hline(y=70, line_dash="dash", line_color="#6366f1",
                          annotation_text="Shortlist Threshold (70%)", annotation_font_color="#6366f1")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            yaxis=dict(range=[0, 105], title="Match Score (%)", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Candidate", showgrid=False),
            margin=dict(t=20, b=20),
            height=320,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Per-candidate deep dive ────────────────────────────────────────
        st.markdown("<div class='section-title'>🔍 Candidate Deep Dives</div>", unsafe_allow_html=True)

        for r in ranked:
            score = r["match_score"]
            label, color = get_match_label(score)
            with st.expander(f"{'✅' if r.get('shortlisted') else '❌'} {r['name']} — {score*100:.1f}%", expanded=(r == ranked[0])):
                dc1, dc2 = st.columns([1, 2])

                with dc1:
                    st.plotly_chart(make_gauge(score, "Match Score"), use_container_width=True)

                with dc2:
                    st.markdown(f"**📧 Email**: {r.get('email') or 'N/A'}")
                    st.markdown(f"**🎓 Education**: {', '.join(r.get('education', [])) or 'N/A'}")
                    st.markdown(f"**📄 File**: `{r['filename']}`")
                    st.markdown(f"**🧮 Validation**: {r['validation']['reason']}")

                    if r.get("skills"):
                        st.markdown("**🛠️ Skills Detected:**")
                        render_skill_tags(r["skills"])

                # Skill gap for this candidate
                gap = analyze_gap(r.get("skills", []), target_role)
                sg1, sg2 = st.columns(2)
                with sg1:
                    if gap["matched_skills"]:
                        st.markdown("**✅ Matched Skills:**")
                        render_skill_tags(gap["matched_skills"])
                with sg2:
                    if gap["missing_skills"]:
                        st.markdown("**❌ Missing Skills:**")
                        render_skill_tags(gap["missing_skills"], missing=True)

                # AI explanation
                with st.spinner("✨ Generating AI explanation..."):
                    explanation = explain_match(
                        candidate_name=r["name"],
                        resume_text=r["text"],
                        job_description=jd_text,
                        match_score=score,
                        candidate_skills=r.get("skills", []),
                        job_title=target_role.replace("-", " ").title(),
                        missing_skills=gap["missing_skills"],
                    )
                st.markdown("**🤖 AI Explanation:**")
                st.markdown(explanation)

                # Save to DB
                db.save_screening(
                    filename=r["filename"],
                    candidate=r["name"],
                    skills=r.get("skills", []),
                    match_score=score,
                    job_title=target_role,
                    mode="recruiter",
                    is_valid=r["validation"]["is_valid"],
                )


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.8rem; padding-bottom:1rem;'>
    🧠 <b>ResumeIQ</b> &nbsp;·&nbsp; Intelligent Resume Screening & Job Recommendation System
    &nbsp;·&nbsp; Built with BERT Embeddings · spaCy · Gemini AI · JSearch API
</div>
""", unsafe_allow_html=True)
