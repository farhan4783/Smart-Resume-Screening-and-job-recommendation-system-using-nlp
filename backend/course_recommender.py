"""
backend/course_recommender.py
------------------------------
Maps skill gaps identified by skill_gap.py to online learning resources.
Loads course_map.json (skill → [{platform, title, url}]).
"""

import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAP_PATH = os.path.join(_HERE, "..", "course_map.json")

_course_map: dict[str, list[dict]] = {}


def _load_map():
    global _course_map
    if _course_map:
        return
    try:
        with open(_MAP_PATH, encoding="utf-8") as f:
            _course_map = json.load(f)
    except FileNotFoundError:
        print(f"[course_recommender] course_map.json not found at {_MAP_PATH}")


def recommend_courses(missing_skills: list[str]) -> list[dict]:
    """
    Return course recommendations for a list of missing skills.

    Args:
        missing_skills: List of skill strings (title-cased OK)

    Returns:
        List of dicts: {skill, courses: [{platform, title, url}]}
    """
    _load_map()
    recommendations = []

    for skill in missing_skills:
        # Try exact match first, then case-insensitive
        courses = _course_map.get(skill)
        if not courses:
            skill_lower = skill.lower()
            for key, val in _course_map.items():
                if key.lower() == skill_lower:
                    courses = val
                    break
        if not courses:
            # Try partial match (e.g., "Cloud AWS" → "AWS")
            for key, val in _course_map.items():
                if skill_lower in key.lower() or key.lower() in skill_lower:
                    courses = val
                    break

        recommendations.append({
            "skill": skill,
            "courses": courses if courses else [
                {
                    "platform": "Coursera",
                    "title": f"Search '{skill}' on Coursera",
                    "url": f"https://www.coursera.org/search?query={skill.replace(' ', '+')}"
                },
                {
                    "platform": "Udemy",
                    "title": f"Search '{skill}' on Udemy",
                    "url": f"https://www.udemy.com/courses/search/?q={skill.replace(' ', '+')}"
                }
            ]
        })

    return recommendations
