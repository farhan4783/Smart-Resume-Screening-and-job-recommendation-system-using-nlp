"""
backend/skill_gap.py
---------------------
Compares candidate skills against a role-specific skills knowledge base.
Identifies gaps and returns structured results.
"""

import os
import csv
import re

# Resolve path relative to this file
_HERE = os.path.dirname(os.path.abspath(__file__))
_KB_PATH = os.path.join(_HERE, "..", "skills_kb.csv")

_skills_kb: dict[str, list[str]] = {}


def _load_kb():
    global _skills_kb
    if _skills_kb:
        return
    try:
        with open(_KB_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                role = row["Role"].strip().upper()
                skills_raw = row["Required_Skills"]
                skills = [s.strip().lower() for s in skills_raw.split(",") if s.strip()]
                _skills_kb[role] = skills
    except FileNotFoundError:
        print(f"[skill_gap] skills_kb.csv not found at {_KB_PATH}")


def get_all_roles() -> list[str]:
    """Return list of all roles in the knowledge base."""
    _load_kb()
    return sorted(_skills_kb.keys())


def analyze_gap(candidate_skills: list[str], target_role: str) -> dict:
    """
    Identify skill gaps between candidate and a target role.

    Args:
        candidate_skills: List of skills extracted from resume (title-cased OK)
        target_role: Role name (e.g., 'INFORMATION-TECHNOLOGY')

    Returns:
        dict with:
            - matched_skills: skills the candidate already has
            - missing_skills: skills the candidate lacks
            - coverage_pct: float (0–100)
    """
    _load_kb()
    role_key = target_role.strip().upper()

    # Normalize candidate skills to lowercase
    candidate_lower = set([s.lower() for s in candidate_skills])

    required = _skills_kb.get(role_key, [])
    if not required:
        # Fuzzy fallback: find closest role
        for key in _skills_kb:
            if role_key in key or key in role_key:
                required = _skills_kb[key]
                break

    matched = []
    missing = []
    for skill in required:
        skill_lower = skill.lower()
        # Match if skill string appears in any candidate skill or vice versa
        found = any(
            skill_lower in cskill or cskill in skill_lower
            for cskill in candidate_lower
        )
        if found:
            matched.append(skill.title())
        else:
            missing.append(skill.title())

    total = len(required)
    coverage = (len(matched) / total * 100) if total > 0 else 0

    return {
        "role": target_role,
        "matched_skills": matched,
        "missing_skills": missing,
        "required_count": total,
        "coverage_pct": round(coverage, 1),
    }


def infer_role_from_skills(candidate_skills: list[str]) -> str:
    """
    Infer the most likely role based on skill overlap with knowledge base.
    Returns role key string.
    """
    _load_kb()
    if not candidate_skills:
        return "INFORMATION-TECHNOLOGY"

    candidate_lower = set([s.lower() for s in candidate_skills])
    best_role = "INFORMATION-TECHNOLOGY"
    best_overlap = -1

    for role, req_skills in _skills_kb.items():
        overlap = sum(
            1 for sk in req_skills
            if any(sk in cs or cs in sk for cs in candidate_lower)
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_role = role

    return best_role
