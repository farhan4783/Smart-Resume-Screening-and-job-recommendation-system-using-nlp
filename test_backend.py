"""Quick integration test for all refined backend modules."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("TESTING REFINED BACKEND MODULES")
print("=" * 60)

# Test 1: NER
print("\n[1] NER Extractor...")
from backend.ner_extractor import extract_entities
from backend.parser import extract_text
t = extract_text(r"DataSet\data\Resumes\AjayKumar.docx", file_name="AjayKumar.docx")
print("   Text length:", len(t))
e = extract_entities(t)
skills = e["skills"]
exp = e["experience_years"]
jtitles = e["job_titles"]
links = e["links"]
print("   Name:", e["name"])
print("   Skills:", len(skills), "detected")
print("   First 5:", skills[:5])
print("   Experience:", exp, "years")
print("   Job Titles:", jtitles[:3])
print("   Email:", e["email"][:20] if e["email"] else "N/A")
print("   Proficiency levels:", [(s["name"], s["level"]) for s in e["skills_detailed"][:3]])

# Test 2: Category Classifier
print("\n[2] Category Classifier...")
from backend.category_classifier import predict_category
c = predict_category(t)
print("   Predicted:", c["category"], "conf=", c["confidence"])
print("   Top 3:", c["top_3"])

# Test 3: ATS Scorer
print("\n[3] ATS Scorer...")
from backend.ats_scorer import compute_ats_score
a = compute_ats_score(
    text=t, skills=skills, education=e["education"],
    experience_years=exp, email=e["email"], phone=e["phone"],
    name=e["name"],
    links=[links.get("linkedin",""), links.get("github","")]
)
print("   Score:", a["total_score"], "/ 100 (Grade", a["grade"] + ")")
for name_k, data in a["breakdown"].items():
    print("   ", name_k + ":", data["score"], "/", data["max"])
print("   Tips:", a["suggestions"][:2])

# Test 4: Hybrid Matcher
print("\n[4] Hybrid Matcher...")
from backend.matcher import compute_hybrid_score
h = compute_hybrid_score(
    resume_text=t,
    jd_text="Looking for a QA engineer with Selenium automation testing experience, JIRA, and agile methodology",
    candidate_skills=skills,
    required_skills=["selenium", "jira", "agile", "automation testing", "manual testing"],
    category_match=True,
)
print("   Total:", round(h["total_score"]*100, 1), "%")
print("   Semantic:", round(h["semantic_score"]*100, 1), "%")
print("   Skill:", round(h["skill_score"]*100, 1), "%")
print("   Category:", round(h["category_score"]*100, 1), "%")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
