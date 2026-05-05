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

# ─── MASSIVELY EXPANDED TECHNICAL SKILLS VOCABULARY (500+ entries) ──────────────
SKILLS_VOCAB = [
    # Core Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "ruby", "go", "golang", "rust",
    "scala", "kotlin", "swift", "php", "r", "matlab", "perl", "shell", "bash", "powershell",
    "objective-c", "dart", "lua", "groovy", "haskell", "elixir", "clojure", "fortran",
    "assembly", "cobol", "vba", "visual basic", "erlang", "f#", "julia", "ocaml", "nim",
    
    # Web Frontend & UI Libraries
    "react", "angular", "vue", "vue.js", "svelte", "next.js", "nuxt.js", "gatsby", "ember",
    "html", "html5", "css", "css3", "sass", "less", "bootstrap", "tailwind", "tailwind css",
    "material ui", "mui", "chakra ui", "ant design", "styled components", "framer motion",
    "jquery", "webpack", "vite", "babel", "rollup", "parcel", "esbuild",
    "redux", "mobx", "zustand", "recoil", "pinia", "rxjs", "graphql", "apollo",
    "responsive design", "progressive web app", "pwa", "single page application", "spa", "webassembly", "wasm",
    
    # Web Backend & API Frameworks
    "node.js", "nodejs", "deno", "bun", "django", "flask", "fastapi", "spring", "spring boot", "express", "express.js",
    ".net", "asp.net", "asp.net core", "ruby on rails", "rails", "laravel", "symfony", "codeigniter",
    "rest api", "soap", "grpc", "websocket", "microservices", "serverless", "soa", "nestjs",
    "tornado", "koa", "hapi", "phoenix", "fiber", "gin", "echo",
    
    # Mobile Development
    "android", "ios", "react native", "flutter", "xamarin", "ionic", "swiftui",
    "kotlin multiplatform", "cordova", "capacitor",
    
    # Data Science, ML & AI
    "machine learning", "deep learning", "artificial intelligence", "ai", "ml", "dl",
    "nlp", "natural language processing", "computer vision", "reinforcement learning",
    "llm", "large language models", "generative ai", "prompt engineering",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm", "catboost",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly", "bokeh", "d3.js",
    "opencv", "hugging face", "transformers", "bert", "gpt", "llama", "stable diffusion",
    "data science", "data analysis", "data visualization", "statistical analysis",
    "data mining", "feature engineering", "model deployment", "mlops", "langchain",
    
    # Big Data & Data Engineering
    "big data", "spark", "apache spark", "hadoop", "hive", "pig", "airflow", "luigi", "presto", "trino",
    "kafka", "rabbitmq", "activemq", "zeromq", "kinesis", "flink", "storm",
    "tableau", "power bi", "looker", "qlik", "metabase", "dbt",
    "jupyter", "colab", "rstudio", "databricks", "snowflake", "redshift", "bigquery",
    "data warehousing", "etl", "elt", "data pipeline", "data modeling",
    
    # Cloud, DevOps & Infrastructure
    "aws", "amazon web services", "azure", "microsoft azure", "google cloud", "gcp",
    "heroku", "digitalocean", "vercel", "netlify", "cloudflare", "linode",
    "docker", "kubernetes", "k8s", "docker swarm", "openshift",
    "jenkins", "gitlab ci", "github actions", "circleci", "travis ci", "bitbucket pipelines",
    "terraform", "ansible", "puppet", "chef", "vagrant", "pulumi", "cloudformation",
    "linux", "unix", "ubuntu", "centos", "redhat", "debian", "alpine", "posix",
    "nginx", "apache", "haproxy", "traefik", "load balancing", "cdn",
    "ci/cd", "devops", "devsecops", "sre", "site reliability engineering", "infrastructure as code", "iac",
    "prometheus", "grafana", "datadog", "new relic", "splunk", "elk stack", "logstash", "kibana",
    
    # Databases & Storage
    "sql", "mysql", "postgresql", "postgres", "mongodb", "redis", "oracle", "sqlite", "nosql",
    "elasticsearch", "cassandra", "dynamodb", "firebase", "supabase", "cosmos db",
    "neo4j", "couchdb", "mariadb", "mssql", "sql server",
    "memcached", "rocksdb", "cockroachdb", "clickhouse", "timescaledb",
    "orm", "sqlalchemy", "prisma", "hibernate", "mongoose", "typeorm",
    
    # Architecture, Patterns & Concepts
    "object-oriented programming", "oop", "functional programming", "fp",
    "mvc", "mvvm", "solid principles", "design patterns", "test-driven development", "tdd",
    "behavior-driven development", "bdd", "domain-driven design", "ddd",
    "event-driven architecture", "clean architecture", "hexagonal architecture",
    "restful architecture", "distributed systems", "concurrency", "multithreading", "asyncio",
    
    # Version Control, PM & Collaboration
    "git", "github", "gitlab", "bitbucket", "svn", "mercurial",
    "jira", "confluence", "trello", "asana", "notion", "monday.com", "linear",
    "agile", "scrum", "kanban", "waterfall", "lean", "six sigma",
    
    # Design & UX
    "figma", "adobe xd", "sketch", "invision", "zeplin",
    "ui design", "ux design", "user research", "wireframing", "prototyping",
    
    # Testing & QA
    "selenium", "cypress", "playwright", "puppeteer", "jest", "mocha", "chai", "jasmine", "vitest",
    "pytest", "unittest", "junit", "testng", "nunit", "xunit",
    "manual testing", "automation testing", "performance testing", "load testing",
    "api testing", "regression testing", "unit testing", "integration testing", "e2e testing",
    "postman", "soapui", "jmeter", "k6", "locust",
    
    # Cybersecurity & Blockchain
    "cybersecurity", "penetration testing", "ethical hacking", "owasp", "cryptography",
    "encryption", "ssl", "tls", "firewall", "siem", "soc", "iam", "oauth", "jwt",
    "blockchain", "smart contracts", "solidity", "web3", "ethereum", "bitcoin", "ipfs",
    
    # Core Engineering & Hardware
    "autocad", "solidworks", "matlab", "simulink", "ros", "robotics",
    "embedded systems", "iot", "internet of things", "arduino", "raspberry pi",
    "fpga", "verilog", "vhdl", "rtos",
    
    # Soft Skills
    "communication", "leadership", "teamwork", "problem solving", "critical thinking",
    "time management", "mentoring"
]

# Common tech-heavy job titles for extraction
JOB_TITLES = [
    "software engineer", "senior software engineer", "principal software engineer", "staff software engineer",
    "software developer", "web developer", "full stack developer", "full-stack developer", "fullstack developer",
    "frontend developer", "front-end developer", "backend developer", "back-end developer",
    "devops engineer", "site reliability engineer", "sre", "cloud engineer", "cloud architect",
    "data scientist", "senior data scientist", "data analyst", "data engineer", "big data engineer",
    "machine learning engineer", "ml engineer", "ai engineer", "artificial intelligence engineer",
    "nlp engineer", "computer vision engineer", "research scientist", "applied scientist",
    "product manager", "technical product manager", "project manager", "scrum master",
    "business analyst", "systems analyst", "database administrator", "dba", "network engineer",
    "solutions architect", "enterprise architect", "technical lead", "tech lead", "team lead",
    "cto", "chief technology officer", "vp of engineering", "director of engineering",
    "qa engineer", "quality assurance engineer", "test engineer", "sdet", "software development engineer in test",
    "security analyst", "security engineer", "cybersecurity engineer", "penetration tester",
    "ui designer", "ux designer", "ui/ux designer", "product designer",
    "system administrator", "sysadmin", "infrastructure engineer", "release engineer",
    "blockchain developer", "smart contract developer", "web3 developer",
    "embedded software engineer", "hardware engineer", "robotics engineer"
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
