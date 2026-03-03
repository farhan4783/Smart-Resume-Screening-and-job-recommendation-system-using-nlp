"""
backend/llm_explainer.py
-------------------------
Generates plain-language match explanations using Google Gemini 2.0 Flash.
Falls back to a template-based explanation if API key is not set.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _client = genai.GenerativeModel("gemini-2.0-flash")
        return _client
    except Exception as e:
        print(f"[llm_explainer] Failed to init Gemini: {e}")
        return None


def _template_explanation(
    candidate_name: str,
    match_score: float,
    candidate_skills: list[str],
    job_title: str,
    missing_skills: list[str],
) -> str:
    """Generate a rule-based explanation when Gemini is not available."""
    pct = match_score * 100
    matched = [s for s in candidate_skills if s][:5]

    if pct >= 70:
        tone = "strong candidate"
        verdict = "Highly Recommended for shortlisting."
    elif pct >= 50:
        tone = "moderate candidate"
        verdict = "Consider for further evaluation."
    else:
        tone = "early-career candidate"
        verdict = "May need additional screening or upskilling."

    explanation = (
        f"**{candidate_name or 'Candidate'}** is a **{tone}** for the **{job_title}** role "
        f"with a match score of **{pct:.1f}%**.\n\n"
        f"✅ **Strengths**: Strong proficiency in {', '.join(matched) if matched else 'core competencies'}. "
        f"The candidate's background aligns well with the technical requirements of this position.\n\n"
    )
    if missing_skills:
        explanation += (
            f"⚠️ **Gaps Identified**: The candidate lacks experience in "
            f"{', '.join(missing_skills[:5])}. "
            f"Upskilling in these areas would significantly improve the fit.\n\n"
        )
    explanation += f"📋 **Recommendation**: {verdict}"
    return explanation


def explain_match(
    candidate_name: str,
    resume_text: str,
    job_description: str,
    match_score: float,
    candidate_skills: list[str],
    job_title: str,
    missing_skills: list[str],
) -> str:
    """
    Generate a plain-language explanation of why a candidate matches (or doesn't)
    a job description.

    Args:
        candidate_name: Name from NER extraction
        resume_text: Raw resume text (truncated internally)
        job_description: Job description text
        match_score: 0.0–1.0 cosine similarity score
        candidate_skills: Skills extracted from resume
        job_title: Target role
        missing_skills: Skills the candidate is missing

    Returns:
        Markdown-formatted explanation string
    """
    client = _get_client()
    if client is None:
        return _template_explanation(candidate_name, match_score, candidate_skills, job_title, missing_skills)

    skills_str = ", ".join(candidate_skills[:10]) if candidate_skills else "not extracted"
    missing_str = ", ".join(missing_skills[:10]) if missing_skills else "none identified"
    pct = match_score * 100

    prompt = f"""You are an expert career coach and hiring manager. Analyze this resume-to-job match and provide a concise, actionable explanation.

**Candidate**: {candidate_name or "Unknown"}
**Target Role**: {job_title}
**Match Score**: {pct:.1f}%
**Candidate Skills**: {skills_str}
**Missing Skills**: {missing_str}
**Job Description** (excerpt): {job_description[:800]}
**Resume** (excerpt): {resume_text[:800]}

Write a 3-paragraph plain-language explanation:
1. Why this candidate is (or isn't) a good fit — reference specific skills
2. The most critical skill gaps and why they matter for this role
3. A clear recommendation: Shortlist / Consider / Pass, with actionable next steps

Use markdown formatting with bold for key points. Be professional, empathetic, and specific."""

    try:
        response = client.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[llm_explainer] Gemini API error: {e}")
        return _template_explanation(candidate_name, match_score, candidate_skills, job_title, missing_skills)


def generate_resume_feedback(resume_text: str, job_description: str = "") -> str:
    """
    Generate improvement suggestions for an uploaded resume.
    """
    client = _get_client()
    if client is None:
        return (
            "💡 **Resume Improvement Tips**\n\n"
            "1. Quantify your achievements (e.g., 'Reduced load time by 40%')\n"
            "2. Use action verbs: Developed, Led, Optimized, Implemented\n"
            "3. Tailor your skills section to include keywords from the job description\n"
            "4. Keep resume to 1–2 pages with clear section headings\n"
            "5. Include links to GitHub, LinkedIn, or portfolio projects"
        )

    jd_context = f"\n\nTarget Job Description: {job_description[:500]}" if job_description else ""
    prompt = f"""You are a professional resume coach. Review the following resume and provide 5 specific, actionable improvement suggestions. Focus on structure, impact statements, keyword optimization, and ATS compatibility.

Resume (excerpt):
{resume_text[:1500]}
{jd_context}

Format your response as a numbered list with bold headers for each suggestion. Be specific and practical."""

    try:
        response = client.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[llm_explainer] Feedback generation error: {e}")
        return "Unable to generate AI feedback at this time. Please check your Gemini API key."
