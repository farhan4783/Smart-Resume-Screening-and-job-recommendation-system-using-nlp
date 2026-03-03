"""
backend/ner_extractor.py
------------------------
Named Entity Recognition using spaCy + regex patterns.
Extracts: Name, Email, Phone, Skills, Education, Organizations.
"""

import re
import spacy

# Skills vocabulary — curated list of common technical and soft skills
SKILLS_VOCAB = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "ruby", "go", "rust",
    "scala", "kotlin", "swift", "php", "r", "matlab", "perl", "shell", "bash", "powershell",
    # Web / Frameworks
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring", "express",
    "html", "css", "sass", "bootstrap", "tailwind", "next.js", "graphql", "rest api", "soap",
    # Data / ML
    "machine learning", "deep learning", "nlp", "natural language processing", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
    "tableau", "power bi", "data analysis", "data visualization", "statistical analysis",
    # Cloud / DevOps
    "aws", "azure", "google cloud", "gcp", "docker", "kubernetes", "jenkins", "ci/cd",
    "terraform", "ansible", "linux", "unix", "devops", "microservices", "kafka",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "oracle", "sqlite", "nosql",
    "elasticsearch", "cassandra", "dynamodb",
    # Tools / Productivity
    "git", "github", "jira", "confluence", "agile", "scrum", "kanban", "ms office",
    "excel", "powerpoint", "word", "slack", "trello", "figma", "adobe xd", "photoshop",
    # Business / Management
    "project management", "business analysis", "stakeholder management", "risk management",
    "financial modeling", "forecasting", "budgeting", "crm", "salesforce", "erp", "sap",
    # Healthcare
    "hipaa", "emr", "ehr", "icd-10", "medical coding", "patient care", "clinical",
    # Other
    "autocad", "solidworks", "catia", "plc", "scada", "six sigma", "lean", "pmp",
]

# Degree keywords
DEGREE_KEYWORDS = [
    r"b\.?sc?\.?\b", r"b\.?tech\b", r"b\.?e\.?\b", r"bachelor", r"b\.a\.?\b",
    r"m\.?sc?\.?\b", r"m\.?tech\b", r"m\.?e\.?\b", r"master", r"m\.b\.a\.?\b", r"mba\b",
    r"ph\.?d\.?\b", r"doctorate", r"associate", r"diploma",
]
DEGREE_PATTERN = re.compile(
    r"(" + "|".join(DEGREE_KEYWORDS) + r")",
    re.IGNORECASE
)

EMAIL_PATTERN = re.compile(r"[\w\.\+\-]+@[\w\-]+\.[\w\.-]+")
PHONE_PATTERN = re.compile(r"(\+?\d[\d\s\-\(\)\.]{7,}\d)")


def _load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        return spacy.load("en_core_web_sm")


_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = _load_spacy_model()
    return _nlp


def extract_email(text: str) -> str:
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = PHONE_PATTERN.search(text)
    if match:
        phone = re.sub(r"\s+", " ", match.group(0)).strip()
        return phone
    return ""


def extract_skills(text: str) -> list[str]:
    """Match skills vocabulary against resume text (case-insensitive)."""
    text_lower = text.lower()
    found = []
    for skill in SKILLS_VOCAB:
        # Use word-boundary-aware check
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill.title() if " " not in skill else skill.upper() if len(skill) <= 4 else skill.title())
    return list(dict.fromkeys(found))  # deduplicate preserving order


def extract_education(text: str) -> list[str]:
    """Extract degree mentions."""
    degrees = DEGREE_PATTERN.findall(text)
    return list(set([d.strip() for d in degrees if d.strip()]))


def extract_entities(text: str, anonymize_pii: bool = False) -> dict:
    """
    Main NER extraction function.

    Args:
        text: Raw resume text
        anonymize_pii: If True, replaces name/email/phone with [REDACTED]

    Returns:
        dict with keys: name, email, phone, skills, education, organizations
    """
    nlp = _get_nlp()

    # Limit text to first 50,000 chars to keep spaCy fast
    doc = nlp(text[:50000])

    # --- Name: first PERSON entity in the doc ---
    name = ""
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not name:
            name = ent.text.strip()
            break

    # --- Organizations ---
    orgs = list(dict.fromkeys([
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in ("ORG",) and len(ent.text.strip()) > 2
    ]))[:10]

    email = extract_email(text)
    phone = extract_phone(text)
    skills = extract_skills(text)
    education = extract_education(text)

    if anonymize_pii:
        name = "[REDACTED]"
        email = "[REDACTED]"
        phone = "[REDACTED]"

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "education": education,
        "organizations": orgs,
    }
