"""
backend/validator.py
---------------------
Heuristic resume validation scoring.
Scores a document against resume-signal keywords and penalizes academic markers.

Returns:
    score (int 0–100): Higher = more likely a valid resume
    is_valid (bool): True if score >= 45
    reason (str): Human-readable explanation
"""

import re

# Keywords that strongly indicate a resume
RESUME_SIGNALS = [
    "experience", "skills", "education", "work history", "employment",
    "responsibilities", "projects", "certifications", "objective", "summary",
    "proficient", "developed", "managed", "achieved", "designed", "implemented",
    "bachelor", "master", "degree", "university", "college", "gpa",
    "years of experience", "references", "profile", "career"
]

# Keywords that indicate an academic paper (penalty)
ACADEMIC_PENALTIES = [
    "abstract", "conclusion", "introduction", "methodology", "literature review",
    "hypothesis", "theorem", "appendix", "bibliography", "citations",
    "figure 1", "table 1", "doi", "journal", "proceedings", "published in",
    "references section", "keywords:", "ieee", "elsevier"
]

# Keywords that indicate spam / non-resume content (heavy penalty)
SPAM_SIGNALS = [
    "click here", "subscribe", "free offer", "discount", "invoice", "payment due"
]


def validate_resume(text: str) -> dict:
    """
    Validate whether a document is a legitimate resume.

    Args:
        text: Raw text extracted from document

    Returns:
        dict with keys: score (int), is_valid (bool), reason (str)
    """
    text_lower = text.lower()
    word_count = len(text.split())
    score = 0
    found_signals = []
    found_penalties = []

    # --- Too short to be a resume ---
    if word_count < 50:
        return {"score": 5, "is_valid": False, "reason": "Document too short (< 50 words)"}

    # --- Base score for sufficient length ---
    if word_count >= 150:
        score += 10
    if word_count >= 300:
        score += 5

    # --- Check resume signals (+4 each, max 60) ---
    for kw in RESUME_SIGNALS:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            score += 4
            found_signals.append(kw)
    score = min(score, 75)   # cap signal contribution

    # --- Check academic penalties (-8 each) ---
    for kw in ACADEMIC_PENALTIES:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            score -= 8
            found_penalties.append(kw)

    # --- Check spam/noise (-15 each) ---
    for kw in SPAM_SIGNALS:
        if kw in text_lower:
            score -= 15

    # --- Bonus: email/phone-like pattern (strong resume signal) ---
    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text):
        score += 10
    if re.search(r"\+?\d[\d\s\-\(\)]{8,}\d", text):
        score += 5

    score = max(0, min(100, score))  # clamp 0–100

    is_valid = score >= 45

    # Build reason string
    if is_valid:
        reason = f"Valid resume (score {score}). Detected: {', '.join(found_signals[:5]) or 'resume keywords'}"
    else:
        reason = f"Low confidence document (score {score})."
        if found_penalties:
            reason += f" Academic markers found: {', '.join(found_penalties[:3])}."
        if not found_signals:
            reason += " No resume keywords detected."

    return {"score": score, "is_valid": is_valid, "reason": reason}
