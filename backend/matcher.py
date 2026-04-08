"""
backend/matcher.py
-------------------
Hybrid resume-to-job-description matching engine.
Combines: semantic similarity (BERT) + skill overlap + category alignment.
"""

import numpy as np
from backend.embedder import encode, encode_batch


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two text strings.
    Returns float between 0.0 and 1.0
    """
    vec1 = encode(text1)
    vec2 = encode(text2)
    similarity = float(np.dot(vec1, vec2))
    return max(0.0, min(1.0, similarity))


def compute_skill_overlap(candidate_skills: list, required_skills: list) -> float:
    """
    Compute Jaccard-like skill overlap score.

    Args:
        candidate_skills: Skills extracted from resume (list of str or dict)
        required_skills: Skills from JD or role KB (list of str)

    Returns:
        float 0.0–1.0
    """
    if not required_skills:
        return 0.5  # Neutral when no requirements specified

    # Normalize to lowercase strings
    def normalize(skills):
        result = set()
        for s in skills:
            if isinstance(s, dict):
                result.add(s.get("name", "").lower())
            else:
                result.add(str(s).lower())
        return result

    cand_set = normalize(candidate_skills)
    req_set = normalize(required_skills)

    if not req_set:
        return 0.5

    # Flexible matching: substring containment
    matched = 0
    for req in req_set:
        for cand in cand_set:
            if req in cand or cand in req:
                matched += 1
                break

    return matched / len(req_set)


def compute_hybrid_score(
    resume_text: str,
    jd_text: str,
    candidate_skills: list = None,
    required_skills: list = None,
    category_match: bool = False,
    weights: dict = None,
) -> dict:
    """
    Compute a weighted hybrid match score.

    Components:
        - Semantic similarity (BERT cosine): 60% default
        - Skill overlap: 25% default
        - Category alignment bonus: 15% default

    Args:
        resume_text: Full resume text
        jd_text: Job description text
        candidate_skills: Skills from resume
        required_skills: Skills from JD or role knowledge base
        category_match: Whether predicted category aligns with JD role
        weights: Optional {semantic, skill, category} weight overrides

    Returns:
        dict with total_score, semantic_score, skill_score, category_score, label
    """
    w = weights or {"semantic": 0.60, "skill": 0.25, "category": 0.15}

    # 1. Semantic similarity
    semantic = compute_similarity(resume_text, jd_text)

    # 2. Skill overlap
    skill = compute_skill_overlap(candidate_skills or [], required_skills or [])

    # 3. Category alignment
    cat_score = 1.0 if category_match else 0.3

    # Weighted total
    total = (
        w["semantic"] * semantic +
        w["skill"] * skill +
        w["category"] * cat_score
    )
    total = max(0.0, min(1.0, total))

    return {
        "total_score": round(total, 4),
        "semantic_score": round(semantic, 4),
        "skill_score": round(skill, 4),
        "category_score": round(cat_score, 4),
        "match_pct": f"{total * 100:.1f}%",
    }


def rank_candidates(candidates: list[dict], jd_text: str, required_skills: list = None) -> list[dict]:
    """
    Rank multiple candidates against a single job description using hybrid scoring.

    Args:
        candidates: List of dicts with keys: name, text, skills, (optional) predicted_category
        jd_text: Job description string
        required_skills: Skills required for the role (from KB)

    Returns:
        Sorted list (desc) with added match scores.
    """
    if not candidates:
        return []

    jd_vec = encode(jd_text)
    resume_texts = [c["text"] for c in candidates]
    resume_vectors = encode_batch(resume_texts)

    results = []
    for i, candidate in enumerate(candidates):
        # Semantic
        semantic = float(np.dot(resume_vectors[i], jd_vec))
        semantic = max(0.0, min(1.0, semantic))

        # Skill overlap
        cand_skills = candidate.get("skills", [])
        req_skills = required_skills or []
        skill_overlap = compute_skill_overlap(cand_skills, req_skills)

        # Category match
        pred_cat = candidate.get("predicted_category", "").upper()
        # Simple heuristic: category matches if it's mentioned in JD
        jd_upper = jd_text.upper()
        cat_match = pred_cat in jd_upper or any(
            word in jd_upper for word in pred_cat.split("-")
        ) if pred_cat and pred_cat != "UNKNOWN" else False

        cat_score = 1.0 if cat_match else 0.3

        total = 0.60 * semantic + 0.25 * skill_overlap + 0.15 * cat_score
        total = max(0.0, min(1.0, total))

        results.append({
            **candidate,
            "match_score": round(total, 4),
            "semantic_score": round(semantic, 4),
            "skill_overlap": round(skill_overlap, 4),
            "category_bonus": cat_match,
            "match_pct": f"{total * 100:.1f}%",
            "shortlisted": total >= 0.70,
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results


def get_match_label(score: float) -> tuple[str, str]:
    """Returns a (label, color) tuple for UI display."""
    pct = score * 100
    if pct >= 70:
        return "Strong Match ✅", "#22c55e"
    elif pct >= 50:
        return "Moderate Match ⚠️", "#f59e0b"
    else:
        return "Low Match ❌", "#ef4444"
