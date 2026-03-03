"""
backend/matcher.py
-------------------
Cosine similarity-based resume-to-job-description matching.
Uses pre-normalized embeddings from embedder.py so similarity = dot product.
"""

import numpy as np
from backend.embedder import encode, encode_batch


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two text strings.

    Args:
        text1: Resume text (or any document)
        text2: Job description text

    Returns:
        float between 0.0 and 1.0
    """
    vec1 = encode(text1)
    vec2 = encode(text2)
    # Vectors are already normalized → dot product = cosine similarity
    similarity = float(np.dot(vec1, vec2))
    return max(0.0, min(1.0, similarity))


def rank_candidates(candidates: list[dict], jd_text: str) -> list[dict]:
    """
    Rank multiple candidates against a single job description.

    Args:
        candidates: List of dicts with at least keys:
            - 'name' (str): candidate identifier
            - 'text' (str): resume text
        jd_text: Job description string

    Returns:
        Sorted list (desc) of dicts with added 'match_score' (float 0–1)
        and 'match_pct' (str, formatted percentage).
    """
    if not candidates:
        return []

    jd_vec = encode(jd_text)
    resume_texts = [c["text"] for c in candidates]
    resume_vectors = encode_batch(resume_texts)

    results = []
    for i, candidate in enumerate(candidates):
        score = float(np.dot(resume_vectors[i], jd_vec))
        score = max(0.0, min(1.0, score))
        results.append({
            **candidate,
            "match_score": score,
            "match_pct": f"{score * 100:.1f}%",
            "shortlisted": score >= 0.70,
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results


def get_match_label(score: float) -> tuple[str, str]:
    """
    Returns a (label, color) tuple for UI display.
    score: 0.0–1.0
    """
    pct = score * 100
    if pct >= 70:
        return "Strong Match ✅", "#22c55e"
    elif pct >= 50:
        return "Moderate Match ⚠️", "#f59e0b"
    else:
        return "Low Match ❌", "#ef4444"
