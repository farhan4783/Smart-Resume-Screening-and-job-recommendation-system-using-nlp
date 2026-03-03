"""
backend/job_fetcher.py
-----------------------
Fetches live job listings from JSearch API via RapidAPI.
Falls back to mock data when API key is not configured.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
JSEARCH_HOST = "jsearch.p.rapidapi.com"
CACHE: dict = {}  # Simple in-memory cache keyed by (query, location)
CACHE_TTL = 3600  # Cache for 1 hour


def _get_mock_jobs(job_title: str, skills: list[str]) -> list[dict]:
    """Return mock job listings for demo/testing without API key."""
    skills_str = ", ".join(skills[:5]) if skills else "relevant skills"
    return [
        {
            "job_id": "mock_1",
            "job_title": f"Senior {job_title}",
            "employer_name": "TechCorp Solutions",
            "employer_logo": None,
            "job_employment_type": "FULLTIME",
            "job_city": "Bangalore",
            "job_country": "IN",
            "job_posted_at": "1 day ago",
            "job_min_salary": 1200000,
            "job_max_salary": 1800000,
            "job_salary_currency": "INR",
            "job_description": f"We are looking for an experienced {job_title} with strong {skills_str} skills to join our growing team.",
            "job_apply_link": "https://example.com/apply/1",
        },
        {
            "job_id": "mock_2",
            "job_title": f"{job_title} Engineer",
            "employer_name": "InnovateTech Pvt Ltd",
            "employer_logo": None,
            "job_employment_type": "FULLTIME",
            "job_city": "Hyderabad",
            "job_country": "IN",
            "job_posted_at": "3 days ago",
            "job_min_salary": 900000,
            "job_max_salary": 1400000,
            "job_salary_currency": "INR",
            "job_description": f"Exciting opportunity for a {job_title} professional. Required: {skills_str}.",
            "job_apply_link": "https://example.com/apply/2",
        },
        {
            "job_id": "mock_3",
            "job_title": f"Lead {job_title}",
            "employer_name": "Global Systems Inc",
            "employer_logo": None,
            "job_employment_type": "FULLTIME",
            "job_city": "Mumbai",
            "job_country": "IN",
            "job_posted_at": "5 days ago",
            "job_min_salary": 1500000,
            "job_max_salary": 2500000,
            "job_salary_currency": "INR",
            "job_description": f"Lead {job_title} role requiring expertise in {skills_str} and team leadership.",
            "job_apply_link": "https://example.com/apply/3",
        },
        {
            "job_id": "mock_4",
            "job_title": f"Junior {job_title}",
            "employer_name": "StartupXYZ",
            "employer_logo": None,
            "job_employment_type": "FULLTIME",
            "job_city": "Pune",
            "job_country": "IN",
            "job_posted_at": "2 days ago",
            "job_min_salary": 600000,
            "job_max_salary": 900000,
            "job_salary_currency": "INR",
            "job_description": f"Great entry-level opportunity! Looking for {job_title} with knowledge of {skills_str}.",
            "job_apply_link": "https://example.com/apply/4",
        },
        {
            "job_id": "mock_5",
            "job_title": f"{job_title} Consultant",
            "employer_name": "Deloitte India",
            "employer_logo": None,
            "job_employment_type": "CONTRACTOR",
            "job_city": "Delhi",
            "job_country": "IN",
            "job_posted_at": "1 week ago",
            "job_min_salary": 2000000,
            "job_max_salary": 3500000,
            "job_salary_currency": "INR",
            "job_description": f"Consulting role for experienced {job_title}. Key skills: {skills_str}.",
            "job_apply_link": "https://example.com/apply/5",
        },
    ]


def fetch_jobs(job_title: str, skills: list[str] = None, location: str = "India", num_pages: int = 1) -> list[dict]:
    """
    Fetch live job listings. Uses JSearch API if key is set, else mock data.

    Args:
        job_title: Target job title inferred from resume
        skills: Top skills to refine the query
        location: Search location
        num_pages: Number of result pages (10 results per page)

    Returns:
        List of job dicts with standardized fields.
    """
    skills = skills or []

    if not RAPIDAPI_KEY or RAPIDAPI_KEY == "your_rapidapi_key_here":
        return _get_mock_jobs(job_title, skills)

    query = job_title
    if skills:
        query += " " + " ".join(skills[:3])

    cache_key = (query.lower(), location.lower())
    now = time.time()
    if cache_key in CACHE and (now - CACHE[cache_key]["ts"]) < CACHE_TTL:
        return CACHE[cache_key]["data"]

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": JSEARCH_HOST,
    }

    all_jobs = []
    for page in range(1, num_pages + 1):
        try:
            response = requests.get(
                "https://jsearch.p.rapidapi.com/search",
                headers=headers,
                params={
                    "query": query,
                    "page": str(page),
                    "num_pages": "1",
                    "country": "in",
                    "date_posted": "week",
                },
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json().get("data", [])
                all_jobs.extend(data)
            else:
                print(f"[job_fetcher] API error {response.status_code}: {response.text[:200]}")
                return _get_mock_jobs(job_title, skills)
        except Exception as e:
            print(f"[job_fetcher] Request failed: {e}")
            return _get_mock_jobs(job_title, skills)

    CACHE[cache_key] = {"data": all_jobs, "ts": now}
    return all_jobs if all_jobs else _get_mock_jobs(job_title, skills)
