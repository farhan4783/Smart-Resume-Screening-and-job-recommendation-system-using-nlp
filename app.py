"""
app.py
------
Intelligent Resume Screening & Job Recommendation System
Main Streamlit Application — Refined Edition
"""

# Suppress TensorFlow before any ML library imports
import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
import io
import json
import time
import hashlib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ — Intelligent Resume Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

db.init_db()

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1035 40%, #0d1b2a 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #151530 50%, #0d1b2a 100%);
        border-right: 1px solid rgba(99,102,241,0.15);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }

    /* Hero */
    .hero-header {
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 30%, #c084fc 60%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1rem;
        letter-spacing: 0.01em;
    }

    /* Pipeline Steps */
    .pipeline-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }
    .pipeline-step {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 8px 16px;
        font-size: 0.78rem;
        color: #a5b4fc;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
        transition: all 0.2s;
    }
    .pipeline-step.active {
        background: rgba(99,102,241,0.2);
        border-color: rgba(99,102,241,0.5);
        color: #c7d2fe;
        box-shadow: 0 0 12px rgba(99,102,241,0.2);
    }
    .pipeline-arrow { color: #475569; font-size: 0.9rem; }

    /* Profile Card */
    .profile-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(168,85,247,0.08));
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 24px;
        padding: 2rem;
        position: relative;
        overflow: hidden;
    }
    .profile-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(99,102,241,0.06), transparent 70%);
    }
    .profile-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .profile-role {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 12px;
    }

    /* Category Badge */
    .category-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin: 4px 4px 4px 0;
    }
    .category-badge-outline {
        display: inline-block;
        background: transparent;
        color: #a5b4fc;
        border: 1px solid rgba(99,102,241,0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 500;
        margin: 3px;
    }

    /* Skill Tags */
    .skill-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15));
        color: #c4b5fd;
        border: 1px solid rgba(167,139,250,0.25);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        margin: 3px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .skill-tag:hover {
        background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.25));
        transform: translateY(-1px);
    }
    .skill-tag .dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-expert { background: #22c55e; }
    .dot-proficient { background: #3b82f6; }
    .dot-intermediate { background: #f59e0b; }
    .dot-familiar { background: #94a3b8; }

    .skill-missing {
        display: inline-block;
        background: rgba(239,68,68,0.08);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.2);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        margin: 3px;
        font-weight: 500;
    }

    /* Section Title */
    .section-title {
        color: #e2e8f0;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 1.25rem 0 0.5rem;
        border-left: 3px solid #6366f1;
        padding-left: 0.75rem;
        letter-spacing: -0.01em;
    }

    /* Job Card */
    .job-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.25s;
    }
    .job-card:hover {
        border-color: rgba(99,102,241,0.3);
        box-shadow: 0 4px 24px rgba(99,102,241,0.1);
        transform: translateY(-2px);
    }

    /* ATS Score Ring */
    .ats-ring {
        width: 90px; height: 90px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        margin: 0 auto;
    }
    .ats-score-num {
        font-size: 1.6rem;
        font-weight: 800;
        line-height: 1;
    }
    .ats-grade {
        font-size: 0.7rem;
        font-weight: 600;
        opacity: 0.8;
    }

    /* Metric cards */
    .metric-mini {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .metric-mini-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-mini-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Score breakdown bar */
    .score-bar-container { margin: 4px 0; }
    .score-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-bottom: 3px;
    }
    .score-bar-track {
        height: 6px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    /* Override Streamlit */
    .stTextArea textarea, .stTextInput input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
        border-radius: 12px !important;
    }
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.25s !important;
        letter-spacing: 0.01em !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
    }
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1rem;
    }
    div[data-testid="metric-container"] label { color: #64748b !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #e2e8f0 !important; }

    h1, h2, h3 { color: #e2e8f0 !important; }
    p, li, span { color: #cbd5e1; }
    .stMarkdown p { color: #cbd5e1; }
    hr { border-color: rgba(255,255,255,0.06) !important; }
    .uploadedFile { background: rgba(255,255,255,0.03) !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.15) !important;
        color: #c7d2fe !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def render_skill_tags(skills, missing=False, detailed=False):
    """Render skill tags with optional proficiency dots."""
    if missing:
        tags = " ".join(f'<span class="skill-missing">✗ {s}</span>' for s in skills)
    elif detailed and isinstance(skills, list) and skills and isinstance(skills[0], dict):
        parts = []
        for s in skills:
            level = s.get("level", "Familiar").lower()
            dot_class = f"dot-{level}"
            parts.append(f'<span class="skill-tag"><span class="dot {dot_class}"></span>{s["name"]}</span>')
        tags = " ".join(parts)
    else:
        tags = " ".join(f'<span class="skill-tag">{s}</span>' for s in skills)
    st.markdown(tags, unsafe_allow_html=True)


def score_color(score: float) -> str:
    if score >= 0.70: return "#22c55e"
    if score >= 0.50: return "#f59e0b"
    return "#ef4444"


def make_gauge(score: float, title: str = "Match Score") -> go.Figure:
    pct = score * 100
    color = score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        title={"text": title, "font": {"color": "#94a3b8", "size": 12}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "dtick": 25},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],  "color": "rgba(239,68,68,0.06)"},
                {"range": [50, 70], "color": "rgba(245,158,11,0.06)"},
                {"range": [70, 100],"color": "rgba(34,197,94,0.06)"},
            ],
            "threshold": {
                "line": {"color": "#6366f1", "width": 2},
                "thickness": 0.8, "value": 70
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=35, b=5, l=15, r=15),
        height=200,
        font_color="#e2e8f0",
    )
    return fig


def make_skills_donut(matched: list, missing: list) -> go.Figure:
    """Donut chart for skill coverage."""
    fig = go.Figure(go.Pie(
        values=[len(matched), len(missing)],
        labels=["Matched", "Missing"],
        hole=0.65,
        marker=dict(colors=["#22c55e", "#ef4444"]),
        textinfo="label+value",
        textfont=dict(color="white", size=12),
        hovertemplate="%{label}: %{value} skills<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        margin=dict(t=10, b=10, l=10, r=10),
        height=200,
        showlegend=False,
        annotations=[dict(
            text=f"{len(matched)}/{len(matched)+len(missing)}",
            x=0.5, y=0.5, font_size=18, font_color="#e2e8f0",
            showarrow=False, font_family="Inter"
        )],
    )
    return fig


def render_ats_ring(score: int, grade: str):
    """Render a circular ATS score indicator."""
    if score >= 70:
        border_color = "#22c55e"
        bg = "rgba(34,197,94,0.1)"
    elif score >= 50:
        border_color = "#f59e0b"
        bg = "rgba(245,158,11,0.1)"
    else:
        border_color = "#ef4444"
        bg = "rgba(239,68,68,0.1)"

    st.markdown(f"""
    <div class="ats-ring" style="border: 3px solid {border_color}; background: {bg};">
        <div class="ats-score-num" style="color: {border_color};">{score}</div>
        <div class="ats-grade" style="color: {border_color};">Grade {grade}</div>
    </div>
    <div style="text-align:center; margin-top:6px; font-size:0.72rem; color:#64748b;">ATS Score</div>
    """, unsafe_allow_html=True)


def render_score_breakdown(breakdown: dict):
    """Render ATS score breakdown bars."""
    colors = {
        "Contact Info": "#3b82f6",
        "Skills Section": "#8b5cf6",
        "Experience Detail": "#6366f1",
        "Education": "#a78bfa",
        "Format Quality": "#06b6d4",
        "Keyword Match": "#f59e0b",
        "Online Presence": "#22c55e",
    }
    for name, data in breakdown.items():
        pct = (data["score"] / data["max"]) * 100 if data["max"] > 0 else 0
        color = colors.get(name, "#6366f1")
        st.markdown(f"""
        <div class="score-bar-container">
            <div class="score-bar-label">
                <span>{name}</span>
                <span>{data['score']}/{data['max']}</span>
            </div>
            <div class="score-bar-track">
                <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_pipeline(active_step: int = 0):
    """Render the NLP pipeline visualization."""
    steps = ["📄 Parse", "🔍 Extract", "🏷️ Classify", "🧮 Embed", "📊 Match", "💡 Recommend"]
    parts = []
    for i, step in enumerate(steps):
        cls = "pipeline-step active" if i <= active_step else "pipeline-step"
        parts.append(f'<div class="{cls}">{step}</div>')
        if i < len(steps) - 1:
            parts.append('<span class="pipeline-arrow">→</span>')
    st.markdown(f'<div class="pipeline-container">{"".join(parts)}</div>', unsafe_allow_html=True)


def process_resume(uploaded_file, anonymize: bool = False) -> dict:
    """Full pipeline: parse → validate → NER → classify → ATS → return results."""
    raw_text = extract_text(uploaded_file, file_name=uploaded_file.name)
    validation = validate_resume(raw_text)
    entities = extract_entities(raw_text, anonymize_pii=anonymize)
    _, clean = preprocess(raw_text)
    category = predict_category(raw_text)
    ats = compute_ats_score(
        text=raw_text,
        skills=entities["skills"],
        education=entities["education"],
        experience_years=entities["experience_years"],
        email=entities["email"],
        phone=entities["phone"],
        name=entities["name"],
        links=[entities["links"].get("linkedin", ""), entities["links"].get("github", "")],
    )

    return {
        "filename": uploaded_file.name,
        "raw_text": raw_text,
        "clean_text": clean,
        "validation": validation,
        "entities": entities,
        "category": category,
        "ats": ats,
    }


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.8rem;">
        <div style="font-size:2.2rem;">🧠</div>
        <div style="font-size:1.1rem; font-weight:700; background: linear-gradient(135deg, #818cf8, #c084fc);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;">ResumeIQ</div>
        <div style="font-size:0.7rem; color:#475569; margin-top:2px;">
            Intelligent Resume Screening v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    mode = st.radio(
        "**Mode**",
        ["🎯 Job Seeker", "🏢 Recruiter"],
        help="Job Seeker: upload your resume & find matching jobs.\nRecruiter: rank multiple candidates against a JD."
    )

    st.markdown("---")
    st.markdown("**⚙️ Settings**")
    anonymize_pii = st.toggle(
        "🙈 Blind Screening",
        value=False,
        help="Hides Name, Email, Phone for bias-free evaluation."
    )

    st.markdown("---")
    st.markdown("**📋 Recent Sessions**")
    history = db.get_history(limit=8)
    if history:
        for h in history[:5]:
            score_str = f"{h['match_score']*100:.0f}%" if h.get("match_score") else "—"
            st.markdown(
                f"<div style='font-size:0.72rem; color:#475569; padding:2px 0;'>"
                f"📄 {h['filename'][:18]}… &nbsp;·&nbsp; {score_str}</div>",
                unsafe_allow_html=True
            )
        if st.button("🗑️ Clear", key="clear_hist", use_container_width=True):
            db.clear_history()
            st.rerun()
    else:
        st.markdown("<div style='font-size:0.72rem; color:#334155;'>No sessions yet</div>",
                     unsafe_allow_html=True)


# ─── HERO ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class='hero-header'>Intelligent Resume Screening<br>& Job Recommendation</div>
<div class='hero-sub'>Powered by BERT Semantic Embeddings · spaCy NER · Random Forest · Gemini AI · JSearch API</div>
""", unsafe_allow_html=True)

render_pipeline(-1)
st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# MODE 1: JOB SEEKER
# ════════════════════════════════════════════════════════════════════════════

if mode == "🎯 Job Seeker":
    st.markdown("### 🎯 Job Seeker Dashboard")

    uploaded = st.file_uploader(
        "📄 Upload Resume (PDF or DOCX)",
        type=["pdf", "docx", "doc"],
        key="seeker_upload",
        help="Your resume is processed locally. Supports PDF and DOCX.",
    )

    if uploaded:
        with st.spinner("🔍 Analyzing resume through NLP pipeline..."):
            result = process_resume(uploaded, anonymize=anonymize_pii)
            entities = result["entities"]
            validation = result["validation"]
            raw_text = result["raw_text"]
            category = result["category"]
            ats = result["ats"]

        render_pipeline(3)

        # ── Validation Banner ────────────────────────────────────────────
        if validation["is_valid"]:
            st.success(f"✅ Valid Resume — Confidence: {validation['score']}/100")
        else:
            st.warning(f"⚠️ {validation['reason']}")

        # ── TABS ─────────────────────────────────────────────────────────
        tab_profile, tab_match, tab_upskill, tab_jobs = st.tabs([
            "📊 Profile & ATS", "🎯 Match Analysis", "📚 Upskill Path", "💼 Live Jobs"
        ])

        # ═══════════════════════════════════════════════════════════════
        # TAB 1: PROFILE & ATS
        # ═══════════════════════════════════════════════════════════════
        with tab_profile:
            col_profile, col_ats = st.columns([2, 1])

            with col_profile:
                # Profile Card
                name_display = entities["name"] or "Candidate"
                job_titles = entities.get("job_titles", [])
                role_display = ", ".join(job_titles[:2]) if job_titles else category["category"].replace("-", " ").title()
                exp_years = entities.get("experience_years", 0)

                st.markdown(f"""
                <div class="profile-card">
                    <div class="profile-name">{name_display}</div>
                    <div class="profile-role">{role_display} · {exp_years}+ years experience</div>
                    <div>
                        <span class="category-badge">{category['category']}</span>
                        <span class="category-badge-outline">Confidence: {category['confidence']*100:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Contact + Education row
                st.markdown("")
                ce1, ce2, ce3 = st.columns(3)
                ce1.markdown(f"""<div class="metric-mini">
                    <div class="metric-mini-value" style="font-size:0.85rem;">📧 {entities['email'] or 'N/A'}</div>
                    <div class="metric-mini-label">Email</div>
                </div>""", unsafe_allow_html=True)
                ce2.markdown(f"""<div class="metric-mini">
                    <div class="metric-mini-value" style="font-size:0.85rem;">📱 {entities['phone'] or 'N/A'}</div>
                    <div class="metric-mini-label">Phone</div>
                </div>""", unsafe_allow_html=True)
                edu_str = ", ".join(entities["education"][:2]) or "N/A"
                ce3.markdown(f"""<div class="metric-mini">
                    <div class="metric-mini-value" style="font-size:0.85rem;">🎓 {edu_str}</div>
                    <div class="metric-mini-label">Education</div>
                </div>""", unsafe_allow_html=True)

                # Links
                links = entities.get("links", {})
                link_parts = []
                if links.get("linkedin"):
                    link_parts.append(f"🔗 [LinkedIn]({links['linkedin']})")
                if links.get("github"):
                    link_parts.append(f"💻 [GitHub]({links['github']})")
                if link_parts:
                    st.markdown(" &nbsp;·&nbsp; ".join(link_parts))

                # Top 3 predicted categories
                if category.get("top_3"):
                    st.markdown("<div class='section-title'>🏷️ Category Prediction (Top 3)</div>", unsafe_allow_html=True)
                    for cat_name, cat_prob in category["top_3"]:
                        pct = cat_prob * 100
                        color = "#6366f1" if cat_prob >= 0.3 else "#475569"
                        st.markdown(f"""
                        <div class="score-bar-container">
                            <div class="score-bar-label"><span>{cat_name}</span><span>{pct:.0f}%</span></div>
                            <div class="score-bar-track">
                                <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

            with col_ats:
                # ATS Score
                render_ats_ring(ats["total_score"], ats["grade"])
                st.markdown("")
                render_score_breakdown(ats["breakdown"])

                # Suggestions
                if ats["suggestions"]:
                    st.markdown("<div class='section-title' style='font-size:0.95rem;'>💡 Quick Fixes</div>", unsafe_allow_html=True)
                    for tip in ats["suggestions"][:3]:
                        st.markdown(f"<div style='font-size:0.75rem; color:#94a3b8; padding:2px 0;'>• {tip}</div>", unsafe_allow_html=True)

            # Skills Section
            st.markdown("<div class='section-title'>🛠️ Skills Detected</div>", unsafe_allow_html=True)
            if entities.get("skills_detailed"):
                render_skill_tags(entities["skills_detailed"], detailed=True)
                st.markdown(f"""
                <div style='margin-top:8px; font-size:0.72rem; color:#475569;'>
                    <span class='dot dot-expert' style='display:inline-block; width:8px; height:8px; border-radius:50%;'></span> Expert &nbsp;
                    <span class='dot dot-proficient' style='display:inline-block; width:8px; height:8px; border-radius:50%;'></span> Proficient &nbsp;
                    <span class='dot dot-intermediate' style='display:inline-block; width:8px; height:8px; border-radius:50%;'></span> Intermediate &nbsp;
                    <span class='dot dot-familiar' style='display:inline-block; width:8px; height:8px; border-radius:50%;'></span> Familiar
                </div>""", unsafe_allow_html=True)
            elif entities["skills"]:
                render_skill_tags(entities["skills"])
            else:
                st.info("No specific skills detected. Try a more detailed resume.")

            # Organizations
            if entities.get("organizations"):
                st.markdown("<div class='section-title'>🏢 Organizations</div>", unsafe_allow_html=True)
                render_skill_tags(entities["organizations"])

        # ═══════════════════════════════════════════════════════════════
        # TAB 2: MATCH ANALYSIS
        # ═══════════════════════════════════════════════════════════════
        with tab_match:
            all_roles = get_all_roles()
            inferred_role = infer_role_from_skills(entities["skills"])

            col_role, col_loc = st.columns([2, 1])
            with col_role:
                selected_role = st.selectbox(
                    "🎯 Target Role",
                    options=all_roles,
                    index=all_roles.index(inferred_role) if inferred_role in all_roles else 0,
                    help="Auto-inferred from your resume. Change to analyze gaps for a different role.",
                )
            with col_loc:
                location = st.text_input("📍 Location", value="India")

            if st.button("🚀 Analyze Match & Find Jobs", key="seeker_analyze", use_container_width=True):
                render_pipeline(5)

                # Skill Gap
                with st.spinner("🔬 Running skill gap analysis..."):
                    gap = analyze_gap(entities["skills"], selected_role)

                col_donut, col_metrics, col_detail = st.columns([1, 1, 2])

                with col_donut:
                    st.plotly_chart(make_skills_donut(gap["matched_skills"], gap["missing_skills"]),
                                   use_container_width=True)

                with col_metrics:
                    st.markdown(f"""
                    <div class="metric-mini" style="margin-bottom:8px;">
                        <div class="metric-mini-value" style="color:#22c55e;">{len(gap['matched_skills'])}</div>
                        <div class="metric-mini-label">Skills Matched</div>
                    </div>
                    <div class="metric-mini" style="margin-bottom:8px;">
                        <div class="metric-mini-value" style="color:#ef4444;">{len(gap['missing_skills'])}</div>
                        <div class="metric-mini-label">Skills Missing</div>
                    </div>
                    <div class="metric-mini">
                        <div class="metric-mini-value" style="color:#6366f1;">{gap['coverage_pct']}%</div>
                        <div class="metric-mini-label">Coverage</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_detail:
                    if gap["matched_skills"]:
                        st.markdown("**✅ You Have:**")
                        render_skill_tags(gap["matched_skills"])
                    if gap["missing_skills"]:
                        st.markdown("**❌ Missing:**")
                        render_skill_tags(gap["missing_skills"], missing=True)

                # Hybrid Score Breakdown
                st.markdown("---")
                job_title_query = selected_role.replace("-", " ").title()
                top_skills = entities["skills"][:5]

                with st.spinner(f"🔎 Fetching {job_title_query} jobs & computing scores..."):
                    jobs = fetch_jobs(job_title_query, top_skills, location=location)

                if jobs:
                    scored_jobs = []
                    for job in jobs[:8]:
                        jd = job.get("job_description", job.get("job_title", ""))
                        if jd:
                            hybrid = compute_hybrid_score(
                                resume_text=raw_text,
                                jd_text=jd,
                                candidate_skills=entities["skills"],
                                required_skills=gap.get("matched_skills", []) + gap.get("missing_skills", []),
                                category_match=(category["category"].upper() == selected_role.upper()),
                            )
                            scored_jobs.append({**job, "_score": hybrid["total_score"],
                                              "_breakdown": hybrid})
                        else:
                            scored_jobs.append({**job, "_score": 0.0, "_breakdown": {}})
                    scored_jobs.sort(key=lambda x: x["_score"], reverse=True)

                    top_job = scored_jobs[0] if scored_jobs else None

                    # Display top match with breakdown
                    st.markdown("<div class='section-title'>📊 Hybrid Score Breakdown (Top Match)</div>",
                              unsafe_allow_html=True)

                    if top_job and top_job.get("_breakdown"):
                        bd = top_job["_breakdown"]
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.markdown(f"""<div class="metric-mini">
                            <div class="metric-mini-value" style="color:{score_color(bd['total_score'])};">{bd['total_score']*100:.0f}%</div>
                            <div class="metric-mini-label">Total Score</div>
                        </div>""", unsafe_allow_html=True)
                        mc2.markdown(f"""<div class="metric-mini">
                            <div class="metric-mini-value" style="color:#818cf8;">{bd['semantic_score']*100:.0f}%</div>
                            <div class="metric-mini-label">Semantic (60%)</div>
                        </div>""", unsafe_allow_html=True)
                        mc3.markdown(f"""<div class="metric-mini">
                            <div class="metric-mini-value" style="color:#a78bfa;">{bd['skill_score']*100:.0f}%</div>
                            <div class="metric-mini-label">Skill Overlap (25%)</div>
                        </div>""", unsafe_allow_html=True)
                        mc4.markdown(f"""<div class="metric-mini">
                            <div class="metric-mini-value" style="color:#c084fc;">{bd['category_score']*100:.0f}%</div>
                            <div class="metric-mini-label">Category (15%)</div>
                        </div>""", unsafe_allow_html=True)

                    # AI Explanation
                    if top_job:
                        st.markdown("<div class='section-title'>🤖 AI Match Explanation</div>",
                                  unsafe_allow_html=True)
                        with st.spinner("✨ Generating AI explanation..."):
                            explanation = explain_match(
                                candidate_name=entities["name"],
                                resume_text=raw_text,
                                job_description=top_job.get("job_description", ""),
                                match_score=top_job["_score"],
                                candidate_skills=entities["skills"],
                                job_title=top_job.get("job_title", job_title_query),
                                missing_skills=gap["missing_skills"],
                            )
                        st.markdown(f"<div class='glass-card'>{explanation}</div>",
                                  unsafe_allow_html=True)

                    # Resume Feedback
                    st.markdown("<div class='section-title'>💡 AI Resume Tips</div>",
                              unsafe_allow_html=True)
                    with st.spinner("📝 Generating feedback..."):
                        feedback = generate_resume_feedback(
                            raw_text,
                            top_job.get("job_description", "") if top_job else ""
                        )
                    st.markdown(f"<div class='glass-card'>{feedback}</div>",
                              unsafe_allow_html=True)

                    # Save
                    db.save_screening(
                        filename=result["filename"],
                        candidate=entities["name"],
                        skills=entities["skills"],
                        match_score=top_job["_score"] if top_job else 0,
                        job_title=job_title_query,
                        mode="seeker",
                        is_valid=validation["is_valid"],
                    )

                    # Store scored_jobs for Jobs tab
                    st.session_state["scored_jobs"] = scored_jobs
                    st.session_state["gap"] = gap
                    st.session_state["job_title_query"] = job_title_query

        # ═══════════════════════════════════════════════════════════════
        # TAB 3: UPSKILL PATH
        # ═══════════════════════════════════════════════════════════════
        with tab_upskill:
            gap_data = st.session_state.get("gap")
            if gap_data and gap_data.get("missing_skills"):
                st.markdown("<div class='section-title'>📚 Recommended Learning Path</div>",
                          unsafe_allow_html=True)
                st.markdown(f"<div style='color:#94a3b8; font-size:0.85rem; margin-bottom:1rem;'>"
                          f"Based on {len(gap_data['missing_skills'])} skill gaps identified for your target role.</div>",
                          unsafe_allow_html=True)

                courses = recommend_courses(gap_data["missing_skills"][:8])
                for item in courses:
                    with st.expander(f"📖 {item['skill']}", expanded=False):
                        for course in item["courses"]:
                            st.markdown(
                                f"🎓 **{course['platform']}** — [{course['title']}]({course['url']})"
                            )
            elif gap_data:
                st.success("🎉 No skill gaps identified — you're fully covered for this role!")
            else:
                st.info("👆 Run the Match Analysis first to see your personalized upskill path.")

        # ═══════════════════════════════════════════════════════════════
        # TAB 4: LIVE JOBS
        # ═══════════════════════════════════════════════════════════════
        with tab_jobs:
            scored_jobs = st.session_state.get("scored_jobs")
            if scored_jobs:
                st.markdown(f"<div class='section-title'>💼 Live Job Matches</div>",
                          unsafe_allow_html=True)

                col_gauge, col_list = st.columns([1, 2])
                with col_gauge:
                    if scored_jobs:
                        st.plotly_chart(make_gauge(scored_jobs[0]["_score"], "Best Match"),
                                      use_container_width=True)

                with col_list:
                    for job in scored_jobs[:6]:
                        score = job["_score"]
                        label, color = get_match_label(score)
                        city = job.get("job_city", "N/A")
                        etype = job.get("job_employment_type", "").replace("_", " ").title()
                        posted = job.get("job_posted_at_datetime_utc", job.get("job_posted_at", ""))
                        if isinstance(posted, str) and "T" in posted:
                            posted = posted.split("T")[0]

                        st.markdown(f"""
                        <div class='job-card'>
                            <div style='display:flex; justify-content:space-between; align-items:center;'>
                                <div>
                                    <div style='font-weight:700; color:#e2e8f0; font-size:0.95rem;'>{job.get('job_title','N/A')}</div>
                                    <div style='color:#64748b; font-size:0.8rem;'>🏢 {job.get('employer_name','N/A')} · 📍 {city} · {etype}</div>
                                    <div style='color:#475569; font-size:0.72rem; margin-top:3px;'>📅 {posted}</div>
                                </div>
                                <div style='text-align:right;'>
                                    <div style='font-size:1.3rem; font-weight:800; color:{color};'>{score*100:.0f}%</div>
                                    <div style='font-size:0.68rem; color:{color};'>{label}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        apply_link = job.get("job_apply_link", "#")
                        st.markdown(f"[🔗 Apply Now]({apply_link})")
            else:
                st.info("👆 Run the Match Analysis to discover live job opportunities.")

    else:
        # Empty state
        st.markdown("""
        <div style='text-align:center; padding: 3rem 2rem;'>
            <div style='font-size: 3.5rem; margin-bottom: 0.8rem; opacity: 0.7;'>📄</div>
            <div style='font-size: 1.1rem; font-weight: 600; color: #64748b; margin-bottom: 0.5rem;'>
                Upload your resume to get started</div>
            <div style='font-size: 0.85rem; color: #475569; max-width: 400px; margin: 0 auto;'>
                Our NLP pipeline will extract your profile, predict your role category,
                compute an ATS score, and find matching jobs — all in seconds.
            </div>
            <div style='margin-top: 1.5rem; display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;'>
                <div class='pipeline-step'>📄 PDF/DOCX</div>
                <span class='pipeline-arrow'>→</span>
                <div class='pipeline-step'>🧠 NLP Analysis</div>
                <span class='pipeline-arrow'>→</span>
                <div class='pipeline-step'>💼 Job Matches</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODE 2: RECRUITER
# ════════════════════════════════════════════════════════════════════════════

elif mode == "🏢 Recruiter":
    st.markdown("### 🏢 Recruiter Dashboard")
    st.markdown("Paste a job description, upload candidate resumes, and rank them by hybrid semantic fit.")

    col_jd, col_upload = st.columns([1, 1])

    with col_jd:
        st.markdown("**📋 Job Description**")
        jd_text = st.text_area(
            label="Paste JD",
            placeholder="e.g. We are looking for a Senior Python Developer with 5+ years in Django, REST APIs, Docker, and AWS...",
            height=260,
            key="recruiter_jd",
            label_visibility="collapsed",
        )

    with col_upload:
        st.markdown("**📂 Candidate Resumes**")
        uploaded_files = st.file_uploader(
            "Upload",
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
                    "experience_years": res["entities"]["experience_years"],
                    "validation": res["validation"],
                    "predicted_category": res["category"]["category"],
                    "ats_score": res["ats"]["total_score"],
                    "ats_grade": res["ats"]["grade"],
                })
            except Exception as e:
                st.warning(f"Could not parse {f.name}: {e}")

        progress_bar.progress(1.0, text="Computing hybrid match scores...")

        # Get required skills from knowledge base
        from backend.skill_gap import analyze_gap as _ag
        _temp_gap = _ag([], target_role)
        req_skills = _temp_gap.get("missing_skills", [])

        with st.spinner("🧮 Computing hybrid semantic similarity scores..."):
            ranked = rank_candidates(candidates, jd_text, required_skills=req_skills)

        progress_bar.empty()
        render_pipeline(5)

        st.markdown("---")
        st.markdown("<div class='section-title'>🏆 Candidate Ranking</div>", unsafe_allow_html=True)

        # Summary metrics
        shortlisted = [r for r in ranked if r.get("shortlisted")]
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.markdown(f"""<div class="metric-mini">
            <div class="metric-mini-value">{len(ranked)}</div>
            <div class="metric-mini-label">Total Candidates</div>
        </div>""", unsafe_allow_html=True)
        sm2.markdown(f"""<div class="metric-mini">
            <div class="metric-mini-value" style="color:#22c55e;">{len(shortlisted)}</div>
            <div class="metric-mini-label">Shortlisted (≥70%)</div>
        </div>""", unsafe_allow_html=True)
        sm3.markdown(f"""<div class="metric-mini">
            <div class="metric-mini-value" style="color:#6366f1;">{ranked[0]['match_score']*100:.1f}%</div>
            <div class="metric-mini-label">Top Score</div>
        </div>""" if ranked else "", unsafe_allow_html=True)
        avg_score = sum(r['match_score'] for r in ranked) / len(ranked) * 100 if ranked else 0
        sm4.markdown(f"""<div class="metric-mini">
            <div class="metric-mini-value">{avg_score:.1f}%</div>
            <div class="metric-mini-label">Avg Score</div>
        </div>""", unsafe_allow_html=True)

        # Ranking table
        table_data = []
        for rank_i, r in enumerate(ranked, 1):
            label, _ = get_match_label(r["match_score"])
            table_data.append({
                "Rank": rank_i,
                "Candidate": r["name"],
                "Category": r.get("predicted_category", "—"),
                "Match Score": f"{r['match_score']*100:.1f}%",
                "ATS": f"{r.get('ats_score', 0)} ({r.get('ats_grade', '—')})",
                "Exp": f"{r.get('experience_years', 0)}y",
                "Skills": len(r.get("skills", [])),
                "Status": "✅" if r.get("shortlisted") else "❌",
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_data, file_name="ranking.csv", mime="text/csv")

        # Score chart
        st.markdown("<div class='section-title'>📊 Score Comparison</div>", unsafe_allow_html=True)
        names_chart = [r["name"][:18] for r in ranked]
        scores_chart = [r["match_score"] * 100 for r in ranked]
        colors_chart = [score_color(r["match_score"]) for r in ranked]

        fig_bar = go.Figure(go.Bar(
            x=names_chart, y=scores_chart,
            marker_color=colors_chart,
            text=[f"{s:.0f}%" for s in scores_chart],
            textposition="auto",
            textfont=dict(color="white", size=11),
        ))
        fig_bar.add_hline(y=70, line_dash="dash", line_color="#6366f1",
                          annotation_text="Threshold 70%", annotation_font_color="#6366f1")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            yaxis=dict(range=[0, 105], title="Score (%)", gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(showgrid=False),
            margin=dict(t=15, b=15), height=280,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Per-candidate deep dives
        st.markdown("<div class='section-title'>🔍 Candidate Deep Dives</div>", unsafe_allow_html=True)

        for r in ranked:
            score = r["match_score"]
            label, color = get_match_label(score)
            with st.expander(
                f"{'✅' if r.get('shortlisted') else '❌'} {r['name']} — {score*100:.1f}% | "
                f"ATS: {r.get('ats_score', 0)} | Cat: {r.get('predicted_category', '—')}",
                expanded=(r == ranked[0])
            ):
                dc1, dc2 = st.columns([1, 2])
                with dc1:
                    st.plotly_chart(make_gauge(score, "Match"), use_container_width=True)
                with dc2:
                    st.markdown(f"**📧 Email**: {r.get('email') or 'N/A'}")
                    st.markdown(f"**🎓 Education**: {', '.join(r.get('education', [])) or 'N/A'}")
                    st.markdown(f"**📊 ATS**: {r.get('ats_score', 0)}/100 (Grade {r.get('ats_grade', '—')})")
                    st.markdown(f"**🏷️ Category**: {r.get('predicted_category', 'N/A')}")
                    st.markdown(f"**⏱️ Experience**: {r.get('experience_years', 0)} years")
                    if r.get("skills"):
                        st.markdown("**🛠️ Skills:**")
                        render_skill_tags(r["skills"])

                # Skill gap for this candidate
                gap = analyze_gap(r.get("skills", []), target_role)
                sg1, sg2 = st.columns(2)
                with sg1:
                    if gap["matched_skills"]:
                        st.markdown("**✅ Matched:**")
                        render_skill_tags(gap["matched_skills"])
                with sg2:
                    if gap["missing_skills"]:
                        st.markdown("**❌ Missing:**")
                        render_skill_tags(gap["missing_skills"], missing=True)

                with st.spinner("✨ AI Analysis..."):
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
<div style='text-align:center; color:#1e293b; font-size:0.75rem; padding-bottom:1rem;'>
    🧠 <b>ResumeIQ</b> v2.0 &nbsp;·&nbsp; BERT + Random Forest + spaCy NER + Gemini AI + JSearch API
    &nbsp;·&nbsp; Hybrid Semantic Matching Engine
</div>
""", unsafe_allow_html=True)
