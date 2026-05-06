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
    """Return 10 realistic mock job listings for demo/testing without API key."""
    skills_str = ", ".join(skills[:5]) if skills else "relevant skills"
    top3 = ", ".join(skills[:3]) if skills else "core technologies"

    return [
        {
            "job_id": "mock_1",
            "job_title": f"Senior {job_title}",
            "employer_name": "Infosys",
            "employer_logo": None,
            "job_employment_type": "FULLTIME",
            "job_city": "Bangalore",
            "job_country": "IN",
            "job_posted_at": "1 day ago",
            "job_min_salary": 1200000,
            "job_max_salary": 1800000,
            "job_salary_currency": "INR",
            "job_description": f"We are hiring a Senior {job_title} to join our enterprise engineering team. You will design and develop scalable solutions using {top3}. Required: 3+ years experience, strong problem-solving skills, knowledge of {skills_str}, and experience with agile methodologies.",
            "job_apply_link": "https://careers.infosys.com",
        },
        {
            "job_id": "mock_2",
            "job_title": f"{job_title}",
            "employer_name": "Wipro Technologies",
            "employer_logo": None,
            "job_employment_type": "FULLTIME",
            "job_city": "Hyderabad",
            "job_country": "IN",
            "job_posted_at": "2 days ago",
            "job_min_salary": 800000,
            "job_max_salary": 1400000,
            "job_salary_currency": "INR",
            "job_description": f"Wipro is looking for a {job_title} to work on cutting-edge projects. Key responsibilities include building robust applications, writing clean code, and collaborating with cross-functional teams. Must have: {skills_str}.",
            "job_apply_link": "https://careers.wipro.com",
        },
        {
            "job_id": "mock_3",
            "job_title": f"Lead {job_title}",
            "employer_name": "Tata Consultancy Services",
            "employer_logo": None,
            "job_employment_type": "FULLTIME",
            "job_city": "Mumbai",
            "job_country": "IN",
            "job_posted_at": "3 days ago",
            "job_min_salary": 1800000,
            "job_max_salary": 2800000,
            "job_salary_currency": "INR",
            "job_description": f"TCS is seeking a Lead {job_title} to architect solutions for global clients. The role involves leading a team of 5-8 engineers, designing system architecture, and ensuring delivery excellence. Expertise required in {skills_str} and distributed systems.",
            "job_apply_link": "https://careers.tcs.com",
        },
        {
            "job_id": "mock_4",
            "job_title": f"Junior {job_title}",
            "employer_name": "Freshworks",
            "employer_logo": None,
            "job_employment_type": "FULLTIME",
            "job_city": "Chennai",
            "job_country": "IN",
            "job_posted_at": "1 day ago",
            "job_min_salary": 500000,
            "job_max_salary": 800000,
            "job_salary_currency": "INR",
            "job_description": f"Great opportunity for freshers and early-career professionals! Join Freshworks as a Junior {job_title}. You will learn and grow while working on real-world products. Basic knowledge of {top3} is required. Training provided.",
            "job_apply_link": "https://careers.freshworks.com",
        },
        {
            "job_id": "mock_5",
            "job_title": f"{job_title} - Remote",
            "employer_name": "Razorpay",
            "employer_logo": None,
            "job_employment_type": "FULLTIME",
            "job_city": "Remote (India)",
            "job_country": "IN",
            "job_posted_at": "4 days ago",
            "job_min_salary": 1500000,
            "job_max_salary": 2500000,
            "job_salary_currency": "INR",
            "job_description": f"Razorpay is hiring a {job_title} for a fully remote position. Build and maintain payment infrastructure used by millions. Requirements: {skills_str}, strong understanding of APIs and microservices architecture.",
            "job_apply_link": "https://razorpay.com/careers",
        },
        {
            "job_id": "mock_6",
            "job_title": f"{job_title} II",
            "employer_name": "Amazon India",
            "employer_logo": None,
            "job_employment_type": "FULLTIME",
            "job_city": "Bangalore",
            "job_country": "IN",
            "job_posted_at": "5 days ago",
            "job_min_salary": 2000000,
            "job_max_salary": 3500000,
            "job_salary_currency": "INR",
            "job_description": f"Join Amazon as a {job_title} II and work on large-scale systems that serve millions of customers. You will own end-to-end feature delivery, write high-quality code, and mentor junior engineers. Proficiency in {skills_str} is essential.",
            "job_apply_link": "https://amazon.jobs",
        },
        {
            "job_id": "mock_7",
            "job_title": f"Associate {job_title}",
            "employer_name": "Accenture",
            "employer_logo": None,
            "job_employment_type": "FULLTIME",
            "job_city": "Pune",
            "job_country": "IN",
            "job_posted_at": "2 days ago",
            "job_min_salary": 600000,
            "job_max_salary": 1000000,
            "job_salary_currency": "INR",
            "job_description": f"Accenture is looking for an Associate {job_title} to deliver innovative solutions for Fortune 500 clients. You should have foundational knowledge in {top3} and a willingness to learn enterprise technologies.",
            "job_apply_link": "https://accenture.com/careers",
        },
        {
            "job_id": "mock_8",
            "job_title": f"Staff {job_title}",
            "employer_name": "Flipkart",
            "employer_logo": None,
            "job_employment_type": "FULLTIME",
            "job_city": "Bangalore",
            "job_country": "IN",
            "job_posted_at": "6 days ago",
            "job_min_salary": 3000000,
            "job_max_salary": 5000000,
            "job_salary_currency": "INR",
            "job_description": f"Flipkart is seeking a Staff {job_title} with 8+ years of experience. You will drive technical strategy, set coding standards, and lead architecture decisions for India's largest e-commerce platform. Deep expertise in {skills_str} required.",
            "job_apply_link": "https://flipkart.com/careers",
        },
        {
            "job_id": "mock_9",
            "job_title": f"{job_title} Intern",
            "employer_name": "Google India",
            "employer_logo": None,
            "job_employment_type": "INTERN",
            "job_city": "Hyderabad",
            "job_country": "IN",
            "job_posted_at": "3 days ago",
            "job_min_salary": 80000,
            "job_max_salary": 150000,
            "job_salary_currency": "INR",
            "job_description": f"Internship opportunity at Google! Work alongside world-class engineers on real products. Ideal for students with coursework in {top3} and strong data structures & algorithms skills.",
            "job_apply_link": "https://careers.google.com",
        },
        {
            "job_id": "mock_10",
            "job_title": f"{job_title} Consultant",
            "employer_name": "Deloitte India",
            "employer_logo": None,
            "job_employment_type": "CONTRACTOR",
            "job_city": "Delhi NCR",
            "job_country": "IN",
            "job_posted_at": "1 week ago",
            "job_min_salary": 2000000,
            "job_max_salary": 3500000,
            "job_salary_currency": "INR",
            "job_description": f"Deloitte is looking for an experienced {job_title} Consultant. You will advise clients on technology strategy and implement solutions using {skills_str}. Excellent communication and consulting skills are a must.",
            "job_apply_link": "https://deloitte.com/careers",
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
