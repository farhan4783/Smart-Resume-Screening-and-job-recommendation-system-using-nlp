"""
backend/ats_scorer.py
----------------------
Structured ATS (Applicant Tracking System) compatibility scoring.
Evaluates resume quality across 7 weighted dimensions with calibrated scoring.
"""

import re


def compute_ats_score(
    text: str,
    skills: list = None,
    education: list = None,
    experience_years: int = 0,
    email: str = "",
    phone: str = "",
    name: str = "",
    links: list = None,
    job_description: str = "",
) -> dict:
    """
    Compute a structured ATS compatibility score.

    Returns:
        dict with:
            - total_score (int 0-100)
            - breakdown (dict of category -> {score, max, details})
            - grade (str: A/B/C/D/F)
            - suggestions (list of improvement tips)
    """
    skills = skills or []
    education = education or []
    links = links or []
    text_lower = text.lower()
    breakdown = {}
    suggestions = []

    # ── 1. Contact Information (10 pts) ─────────────────────────────────
    contact_score = 0
    contact_details = []
    if name and name.strip():
        contact_score += 3
        contact_details.append("✅ Name found")
    else:
        suggestions.append("Add your full name prominently at the top of your resume")
        contact_details.append("❌ Name missing")
    if email and email.strip():
        contact_score += 4
        contact_details.append("✅ Email found")
    else:
        suggestions.append("Include a professional email address")
        contact_details.append("❌ Email missing")
    if phone and phone.strip():
        contact_score += 3
        contact_details.append("✅ Phone found")
    else:
        suggestions.append("Add a contact phone number")
        contact_details.append("❌ Phone missing")
    breakdown["Contact Info"] = {"score": contact_score, "max": 10, "details": contact_details}

    # ── 2. Skills Section (20 pts) ───────────────────────────────────────
    skills_score = 0
    skills_details = []
    num_skills = len(skills)

    # More granular and realistic skill scoring
    if num_skills >= 15:
        skills_score = 18
    elif num_skills >= 10:
        skills_score = 15
    elif num_skills >= 7:
        skills_score = 12
    elif num_skills >= 4:
        skills_score = 8
    elif num_skills >= 2:
        skills_score = 5
    elif num_skills >= 1:
        skills_score = 3
    else:
        suggestions.append("Add a dedicated 'Skills' section with at least 5-10 relevant technical skills")

    skills_details.append(f"{num_skills} skills detected")

    # Check for skills section heading
    has_skills_heading = bool(re.search(
        r"\b(skills|technical\s+skills|core\s+competencies|technologies|tech\s+stack)\b",
        text_lower
    ))
    if has_skills_heading:
        skills_score = min(20, skills_score + 2)
        skills_details.append("✅ Skills section heading found")
    else:
        suggestions.append("Use a clear 'Skills' or 'Technical Skills' heading for ATS parsing")
        skills_details.append("⚠️ No explicit skills section heading")

    breakdown["Skills Section"] = {"score": skills_score, "max": 20, "details": skills_details}

    # ── 3. Experience Detail (20 pts) ────────────────────────────────────
    exp_score = 0
    exp_details = []

    # Technical action verbs
    action_verbs = [
        "developed", "implemented", "managed", "designed", "led", "created",
        "built", "optimized", "analyzed", "delivered", "achieved", "improved",
        "reduced", "increased", "launched", "deployed", "automated", "collaborated",
        "mentored", "presented", "negotiated", "resolved", "streamlined",
        "architected", "engineered", "refactored", "migrated", "scaled",
        "orchestrated", "provisioned", "debugged", "troubleshot", "configured",
        "integrated", "benchmarked", "profiled", "secured", "modernized"
    ]
    verb_count = sum(1 for v in action_verbs if re.search(r"\b" + v + r"\b", text_lower))

    if verb_count >= 10:
        exp_score += 8
    elif verb_count >= 6:
        exp_score += 6
    elif verb_count >= 3:
        exp_score += 4
    elif verb_count >= 1:
        exp_score += 2
    else:
        suggestions.append("Use strong action verbs like 'Developed', 'Implemented', 'Optimized'")
    exp_details.append(f"{verb_count} action verbs found")

    # Quantified achievements
    quant_pattern = r"\d+\s*(%|percent|million|billion|thousand|users|clients|projects|team|years|months)"
    quant_matches = len(re.findall(quant_pattern, text_lower))
    if quant_matches >= 5:
        exp_score += 7
    elif quant_matches >= 3:
        exp_score += 5
    elif quant_matches >= 1:
        exp_score += 3
    else:
        suggestions.append("Quantify achievements (e.g., 'Reduced load time by 40%', 'Managed team of 12')")
    exp_details.append(f"{quant_matches} quantified achievements")

    # Experience years
    if experience_years >= 5:
        exp_score += 5
    elif experience_years >= 3:
        exp_score += 4
    elif experience_years >= 1:
        exp_score += 2
    exp_details.append(f"{experience_years}+ years experience detected")

    breakdown["Experience Detail"] = {"score": min(20, exp_score), "max": 20, "details": exp_details}

    # ── 4. Education (10 pts) ────────────────────────────────────────────
    edu_score = 0
    edu_details = []
    if education:
        if len(education) >= 2:
            edu_score = 7
        else:
            edu_score = 5
        edu_details.append(f"{len(education)} degree(s) detected: {', '.join(education[:3])}")
    else:
        suggestions.append("Include your educational background with degree and institution")
        edu_details.append("No degree detected")

    if re.search(r"\b(education|qualification|academic|degree)\b", text_lower):
        edu_score = min(10, edu_score + 3)
        edu_details.append("✅ Education section heading found")
    else:
        edu_details.append("⚠️ No explicit education section heading")

    breakdown["Education"] = {"score": edu_score, "max": 10, "details": edu_details}

    # ── 5. Format Quality (15 pts) ──────────────────────────────────────
    fmt_score = 0
    fmt_details = []
    word_count = len(text.split())

    # Length check
    if 300 <= word_count <= 1200:
        fmt_score += 5
        fmt_details.append(f"✅ Good length ({word_count} words)")
    elif 200 <= word_count < 300:
        fmt_score += 3
        fmt_details.append(f"⚠️ Slightly short ({word_count} words)")
    elif word_count < 200:
        fmt_score += 1
        suggestions.append(f"Resume is too short ({word_count} words). Aim for 300-800 words.")
        fmt_details.append(f"⚠️ Too short ({word_count} words)")
    else:
        fmt_score += 3
        fmt_details.append(f"⚠️ Long ({word_count} words). Consider condensing to 1-2 pages.")

    # Section headings
    headings = ["experience", "education", "skills", "summary", "objective", "projects", "certifications"]
    found_headings = sum(1 for h in headings if re.search(r"\b" + h + r"\b", text_lower))
    if found_headings >= 5:
        fmt_score += 6
    elif found_headings >= 3:
        fmt_score += 4
    elif found_headings >= 2:
        fmt_score += 2
    else:
        suggestions.append("Use clear section headings: Experience, Education, Skills, Projects")
    fmt_details.append(f"{found_headings} standard section headings found")

    # No academic penalties
    academic_terms = ["abstract", "methodology", "hypothesis", "literature review", "proceedings"]
    academic_count = sum(1 for t in academic_terms if t in text_lower)
    if academic_count == 0:
        fmt_score += 4
        fmt_details.append("✅ No academic paper markers")
    else:
        fmt_score += 1
        fmt_details.append(f"⚠️ {academic_count} academic markers found (may confuse ATS)")

    breakdown["Format Quality"] = {"score": min(15, fmt_score), "max": 15, "details": fmt_details}

    # ── 6. Keyword Density vs JD (15 pts) ────────────────────────────────
    kw_score = 0
    kw_details = []

    if job_description:
        jd_words = set(re.findall(r"\b[a-z]{3,}\b", job_description.lower()))
        resume_words = set(re.findall(r"\b[a-z]{3,}\b", text_lower))
        common_stop = {"the", "and", "for", "are", "with", "that", "this", "from", "has", "have",
                       "will", "been", "were", "not", "but", "they", "one", "all", "can", "had",
                       "you", "our", "their", "which", "each", "about", "into", "more", "other"}
        jd_keywords = jd_words - common_stop
        overlap = resume_words & jd_keywords
        overlap_pct = (len(overlap) / max(len(jd_keywords), 1)) * 100

        if overlap_pct >= 50:
            kw_score = 15
        elif overlap_pct >= 35:
            kw_score = 11
        elif overlap_pct >= 20:
            kw_score = 7
        elif overlap_pct >= 10:
            kw_score = 4
        else:
            suggestions.append("Tailor your resume keywords to match the job description more closely")

        kw_details.append(f"{overlap_pct:.0f}% keyword overlap with JD ({len(overlap)}/{len(jd_keywords)} keywords)")
    else:
        # When no JD is provided, give a moderate base score instead of a flat 8
        # Score based on the resume's own keyword richness
        unique_words = set(re.findall(r"\b[a-z]{4,}\b", text_lower))
        richness = len(unique_words)
        if richness >= 200:
            kw_score = 10
        elif richness >= 120:
            kw_score = 8
        elif richness >= 60:
            kw_score = 6
        else:
            kw_score = 4
        kw_details.append(f"No JD provided — scored on vocabulary richness ({richness} unique words)")

    breakdown["Keyword Match"] = {"score": kw_score, "max": 15, "details": kw_details}

    # ── 7. Online Presence (10 pts) ──────────────────────────────────────
    link_score = 0
    link_details = []
    if links:
        for link in links:
            link_lower = link.lower()
            if "linkedin" in link_lower:
                link_score += 4
                link_details.append("✅ LinkedIn profile")
            elif "github" in link_lower:
                link_score += 4
                link_details.append("✅ GitHub profile")
            elif "leetcode" in link_lower:
                link_score += 3
                link_details.append("✅ LeetCode profile")
            elif "kaggle" in link_lower:
                link_score += 3
                link_details.append("✅ Kaggle profile")
            elif "stackoverflow" in link_lower:
                link_score += 3
                link_details.append("✅ StackOverflow profile")
            elif "dev.to" in link_lower or "medium.com" in link_lower:
                link_score += 2
                link_details.append("✅ Tech Blog link")
            else:
                link_score += 2
                link_details.append(f"✅ Portfolio/website link")
    if link_score == 0:
        suggestions.append("Add LinkedIn, GitHub, or portfolio links to strengthen your online presence")
        link_details.append("❌ No professional links found")
    link_score = min(10, link_score)
    breakdown["Online Presence"] = {"score": link_score, "max": 10, "details": link_details}

    # ── Total ────────────────────────────────────────────────────────────
    total = sum(b["score"] for b in breakdown.values())
    total = max(0, min(100, total))

    if total >= 85:
        grade = "A"
    elif total >= 70:
        grade = "B"
    elif total >= 55:
        grade = "C"
    elif total >= 40:
        grade = "D"
    else:
        grade = "F"

    return {
        "total_score": total,
        "grade": grade,
        "breakdown": breakdown,
        "suggestions": suggestions[:5],
    }
