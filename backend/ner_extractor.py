"""
backend/ner_extractor.py
------------------------
Enhanced Named Entity Recognition using spaCy + regex patterns.
Extracts: Name, Email, Phone, Skills (200+), Education, Organizations,
Experience Years, Job Titles, Links (LinkedIn/GitHub/Portfolio).
"""

import re

# Try to import spaCy — falls back to regex-only NER if unavailable
_SPACY_AVAILABLE = False
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    pass

# ─── EXPANDED SKILLS VOCABULARY (200+ entries) ────────────────────────────────
SKILLS_VOCAB = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "ruby", "go", "rust",
    "scala", "kotlin", "swift", "php", "r", "matlab", "perl", "shell", "bash", "powershell",
    "objective-c", "dart", "lua", "groovy", "haskell", "elixir", "clojure", "fortran",
    "assembly", "cobol", "vba", "visual basic",
    # Web Frontend
    "react", "angular", "vue", "svelte", "next.js", "nuxt.js", "gatsby", "ember",
    "html", "css", "sass", "less", "bootstrap", "tailwind", "material ui", "chakra ui",
    "jquery", "webpack", "vite", "babel", "redux", "mobx", "zustand",
    "responsive design", "progressive web app", "single page application",
    # Web Backend / Frameworks
    "node.js", "django", "flask", "fastapi", "spring", "spring boot", "express",
    ".net", "asp.net", "ruby on rails", "laravel", "symfony", "codeigniter",
    "graphql", "rest api", "soap", "grpc", "websocket", "microservices",
    # Mobile Development
    "android", "ios", "react native", "flutter", "xamarin", "ionic", "swiftui",
    "kotlin multiplatform",
    # Data Science / ML / AI
    "machine learning", "deep learning", "artificial intelligence",
    "nlp", "natural language processing", "computer vision", "reinforcement learning",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "opencv", "hugging face", "transformers", "bert", "gpt",
    "data science", "data analysis", "data visualization", "statistical analysis",
    "data mining", "feature engineering", "model deployment", "mlops",
    "big data", "spark", "hadoop", "hive", "pig", "airflow",
    "tableau", "power bi", "looker", "qlik", "metabase",
    "jupyter", "colab", "rstudio",
    # Cloud / DevOps / Infrastructure
    "aws", "azure", "google cloud", "gcp", "heroku", "digitalocean", "vercel", "netlify",
    "docker", "kubernetes", "jenkins", "gitlab ci", "github actions", "circleci",
    "terraform", "ansible", "puppet", "chef", "vagrant",
    "linux", "unix", "ubuntu", "centos", "redhat",
    "nginx", "apache", "load balancing", "cdn",
    "ci/cd", "devops", "sre", "infrastructure as code",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "oracle", "sqlite", "nosql",
    "elasticsearch", "cassandra", "dynamodb", "firebase", "supabase",
    "neo4j", "couchdb", "mariadb", "mssql", "snowflake", "redshift",
    "data warehousing", "etl", "data pipeline",
    # Tools / Version Control / PM
    "git", "github", "gitlab", "bitbucket", "svn",
    "jira", "confluence", "trello", "asana", "notion", "monday.com",
    "slack", "teams", "zoom",
    "agile", "scrum", "kanban", "waterfall", "lean", "six sigma",
    "pmp", "prince2", "safe",
    # Design / UX
    "figma", "adobe xd", "sketch", "invision", "zeplin",
    "photoshop", "illustrator", "indesign", "after effects", "premiere pro",
    "ui design", "ux design", "user research", "wireframing", "prototyping",
    "accessibility", "design thinking",
    # Testing / QA
    "selenium", "cypress", "jest", "mocha", "pytest", "junit", "testng",
    "manual testing", "automation testing", "performance testing", "load testing",
    "api testing", "regression testing", "unit testing", "integration testing",
    "postman", "soapui", "jmeter",
    # Security
    "cybersecurity", "penetration testing", "ethical hacking", "owasp",
    "encryption", "ssl", "firewall", "siem", "soc",
    "information security", "network security", "vulnerability assessment",
    # Business / Finance / Management
    "project management", "business analysis", "stakeholder management", "risk management",
    "financial modeling", "forecasting", "budgeting", "accounting",
    "crm", "salesforce", "erp", "sap", "oracle erp",
    "ms office", "excel", "powerpoint", "word", "outlook",
    "quickbooks", "tally", "bloomberg",
    "supply chain", "operations management", "strategic planning",
    "digital marketing", "seo", "sem", "google analytics", "content marketing",
    "social media marketing", "email marketing", "copywriting",
    # Healthcare
    "hipaa", "emr", "ehr", "icd-10", "medical coding", "patient care", "clinical",
    "fda", "gmp", "pharmacology", "nursing", "phlebotomy",
    # Engineering / Manufacturing
    "autocad", "solidworks", "catia", "plc", "scada", "matlab", "simulink",
    "mechanical design", "electrical engineering", "civil engineering",
    "finite element analysis", "cfd", "3d printing", "cnc",
    # Soft Skills
    "communication", "leadership", "teamwork", "problem solving", "critical thinking",
    "time management", "negotiation", "presentation", "public speaking",
    "conflict resolution", "adaptability", "mentoring",
]

# Common job titles for extraction
JOB_TITLES = [
    "software engineer", "software developer", "web developer", "full stack developer",
    "frontend developer", "backend developer", "devops engineer", "data scientist",
    "data analyst", "data engineer", "machine learning engineer", "ai engineer",
    "product manager", "project manager", "program manager", "scrum master",
    "business analyst", "systems analyst", "database administrator", "network engineer",
    "cloud architect", "solutions architect", "technical lead", "team lead",
    "cto", "ceo", "cfo", "coo", "vp engineering", "director",
    "qa engineer", "test engineer", "sdet", "automation engineer",
    "security analyst", "security engineer", "consultant", "advisor",
    "ui designer", "ux designer", "graphic designer", "art director",
    "content writer", "technical writer", "copywriter",
    "sales manager", "account manager", "marketing manager", "hr manager",
    "recruiter", "talent acquisition", "operations manager",
    "accountant", "financial analyst", "investment banker", "auditor",
    "teacher", "professor", "instructor", "trainer",
    "nurse", "doctor", "pharmacist", "medical officer",
    "chef", "sous chef", "executive chef",
    "lawyer", "advocate", "legal counsel", "paralegal",
    "civil engineer", "mechanical engineer", "electrical engineer", "chemical engineer",
    "intern", "trainee", "associate", "senior", "principal", "staff",
]

# Degree keywords
DEGREE_KEYWORDS = [
    r"b\.?sc?\.?\b", r"b\.?tech\b", r"b\.?e\.?\b", r"bachelor", r"b\.a\.?\b",
    r"b\.?com\b", r"b\.?ca\b", r"bba\b", r"bbs\b",
    r"m\.?sc?\.?\b", r"m\.?tech\b", r"m\.?e\.?\b", r"master", r"m\.b\.a\.?\b", r"mba\b",
    r"m\.?com\b", r"mca\b", r"m\.a\.\b",
    r"ph\.?d\.?\b", r"doctorate", r"associate", r"diploma", r"postgraduate",
    r"12th", r"10th", r"hsc\b", r"ssc\b",
]
DEGREE_PATTERN = re.compile(
    r"(" + "|".join(DEGREE_KEYWORDS) + r")",
    re.IGNORECASE
)

EMAIL_PATTERN = re.compile(r"[\w\.\+\-]+@[\w\-]+\.[\w\.-]+")
PHONE_PATTERN = re.compile(r"(\+?\d[\d\s\-\(\)\.]{7,}\d)")
LINK_PATTERN = re.compile(r"(https?://[^\s,\)]+|www\.[^\s,\)]+)")
LINKEDIN_PATTERN = re.compile(r"(https?://(?:www\.)?linkedin\.com/\S+)", re.IGNORECASE)
GITHUB_PATTERN = re.compile(r"(https?://(?:www\.)?github\.com/\S+)", re.IGNORECASE)

# Name pattern: lines at the start that look like a person's name
NAME_PATTERN = re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$", re.MULTILINE)

# Experience years patterns
EXP_PATTERNS = [
    re.compile(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)", re.IGNORECASE),
    re.compile(r"(?:experience|exp)\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)", re.IGNORECASE),
    re.compile(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:in|of|as)", re.IGNORECASE),
    re.compile(r"over\s*(\d+)\s*(?:years?|yrs?)", re.IGNORECASE),
]


_nlp = None
_spacy_failed = False


def _get_nlp():
    """Load spaCy model with graceful fallback."""
    global _nlp, _spacy_failed
    if _spacy_failed:
        return None
    if _nlp is not None:
        return _nlp
    if not _SPACY_AVAILABLE:
        _spacy_failed = True
        return None
    try:
        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except Exception as e:
        print(f"[ner_extractor] spaCy load failed ({e}). Using regex-only NER.")
        _spacy_failed = True
        return None


def _extract_name_regex(text: str) -> str:
    """Fallback name extraction using regex heuristics."""
    # Look for name-like patterns in the first 500 chars
    head = text[:500]
    # Try to find a capitalized 2-3 word name at the start of a line
    match = NAME_PATTERN.search(head)
    if match:
        candidate = match.group(1).strip()
        # Reject overly common words that aren't names
        reject = {"the", "and", "for", "experience", "skills", "education",
                  "summary", "objective", "profile", "references", "career"}
        if candidate.lower() not in reject and len(candidate) > 3:
            return candidate
    return ""


def extract_email(text: str) -> str:
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = PHONE_PATTERN.search(text)
    if match:
        phone = re.sub(r"\s+", " ", match.group(0)).strip()
        return phone
    return ""


def extract_links(text: str) -> dict:
    """Extract LinkedIn, GitHub, and other professional links."""
    result = {"linkedin": "", "github": "", "portfolio": []}

    linkedin = LINKEDIN_PATTERN.search(text)
    if linkedin:
        result["linkedin"] = linkedin.group(1).rstrip("/.")

    github = GITHUB_PATTERN.search(text)
    if github:
        result["github"] = github.group(1).rstrip("/.")

    all_links = LINK_PATTERN.findall(text)
    for link in all_links:
        link_clean = link.rstrip("/.")
        if "linkedin" not in link.lower() and "github" not in link.lower():
            if link_clean not in result["portfolio"]:
                result["portfolio"].append(link_clean)

    return result


def extract_skills(text: str) -> list[dict]:
    """
    Match skills vocabulary against resume text (case-insensitive).
    Returns list of dicts with skill name and frequency-based weight.
    """
    text_lower = text.lower()
    found = []
    seen = set()

    for skill in SKILLS_VOCAB:
        pattern = r"\b" + re.escape(skill) + r"\b"
        matches = re.findall(pattern, text_lower)
        if matches and skill not in seen:
            # Weight based on mention frequency (capped)
            count = len(matches)
            if count >= 5:
                weight = 1.0
                level = "Expert"
            elif count >= 3:
                weight = 0.8
                level = "Proficient"
            elif count >= 2:
                weight = 0.6
                level = "Intermediate"
            else:
                weight = 0.4
                level = "Familiar"

            # Normalize display name
            display = skill.title()
            if len(skill) <= 4 and " " not in skill:
                display = skill.upper()
            elif "." in skill:
                display = skill  # Keep Node.js, Next.js as-is

            found.append({"name": display, "weight": weight, "level": level, "mentions": count})
            seen.add(skill)

    # Sort by weight descending
    found.sort(key=lambda x: x["weight"], reverse=True)
    return found


def extract_education(text: str) -> list[str]:
    """Extract degree mentions."""
    degrees = DEGREE_PATTERN.findall(text)
    return list(set([d.strip() for d in degrees if d.strip()]))


def extract_experience_years(text: str) -> int:
    """Extract years of experience from resume text."""
    years = []
    for pattern in EXP_PATTERNS:
        matches = pattern.findall(text)
        for m in matches:
            try:
                y = int(m)
                if 0 < y < 50:  # Sanity check
                    years.append(y)
            except ValueError:
                pass

    return max(years) if years else 0


def extract_job_titles(text: str) -> list[str]:
    """Extract job titles mentioned in the resume."""
    text_lower = text.lower()
    found = []
    for title in JOB_TITLES:
        if re.search(r"\b" + re.escape(title) + r"\b", text_lower):
            found.append(title.title())
    return list(dict.fromkeys(found))[:8]


def extract_entities(text: str, anonymize_pii: bool = False) -> dict:
    """
    Main NER extraction function — enhanced version.

    Args:
        text: Raw resume text
        anonymize_pii: If True, replaces name/email/phone with [REDACTED]

    Returns:
        dict with keys: name, email, phone, skills, skills_list, education,
        organizations, experience_years, job_titles, links
    """
    nlp = _get_nlp()

    name = ""
    orgs = []

    if nlp is not None:
        doc = nlp(text[:50000])
        # --- Name: first PERSON entity ---
        for ent in doc.ents:
            if ent.label_ == "PERSON" and not name:
                candidate = ent.text.strip()
                words = candidate.split()
                if 1 <= len(words) <= 5 and not any(c.isdigit() for c in candidate):
                    name = candidate
                    break
        # --- Organizations ---
        orgs = list(dict.fromkeys([
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ == "ORG" and len(ent.text.strip()) > 2
        ]))[:10]
    else:
        # Regex fallback for name
        name = _extract_name_regex(text)

    email = extract_email(text)
    phone = extract_phone(text)
    links = extract_links(text)
    skills_detailed = extract_skills(text)
    skills_names = [s["name"] for s in skills_detailed]
    education = extract_education(text)
    experience_years = extract_experience_years(text)
    job_titles = extract_job_titles(text)

    if anonymize_pii:
        name = "[REDACTED]"
        email = "[REDACTED]"
        phone = "[REDACTED]"

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills_names,           # Backward compatible: list of strings
        "skills_detailed": skills_detailed, # New: list of {name, weight, level, mentions}
        "education": education,
        "organizations": orgs,
        "experience_years": experience_years,
        "job_titles": job_titles,
        "links": links,
    }
