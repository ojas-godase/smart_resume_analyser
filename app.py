from flask import Flask, request, render_template
import fitz  
import pandas as pd
import re
import os
from dotenv import load_dotenv
import json
# from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import tempfile

load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1")

# Semantic similarity model for JD ranking
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Utility Functions ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join([page.get_text() for page in doc])

# --- ATS Score Calculation ---

SECTION_PATTERNS = {
    "Education": r"(?i)\b(education|academic background)\b",
    "Experience": r"(?i)\b(work experience|professional experience|employment history|experience)\b",
    "Projects": r"(?i)\b(projects|personal projects)\b",
    "Skills": r"(?i)\b(skills|technical skills|key skills)\b",
    "Certifications": r"(?i)\b(certifications|licenses)\b",
    "Summary": r"(?i)\b(career objective|summary|professional summary|objective)\b",
    "Contact": r"(?i)\b(phone|email|linkedin|github)\b",
    "Achievements": r"(?i)\b(achievements|awards|honors)\b",
    "Languages": r"(?i)\b(languages spoken|languages)\b",
    "Tools": r"(?i)\b(technologies|tools|software)\b"
}

def analyze_sections(text):
    """
    Returns dict with each section:
    { "Education": {"present": True, "bullet_count": 3}, ... }
    """
    section_stats = {}
    for section, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text)
        present = bool(match)
        bullet_count = 0
        if present:
            section_text = text[match.start():match.start() + 1000]
            bullet_count = len(re.findall(r"[\n•\-‣▪▶●][ \t]*", section_text))
        section_stats[section] = {"present": present, "bullet_count": bullet_count}
    return section_stats


def calculate_ats_score(text):
    """
    ATS score (0–100) using section presence + resume heuristics.
    """

    if not text:
        return 0

    lower = text.lower()
    score = 0
    total = 12  

    sections = analyze_sections(text)

    # --- Core sections (weighted more) ---
    if sections["Experience"]["present"]:
        score += 2
    if sections["Education"]["present"]:
        score += 2
    if sections["Skills"]["present"]:
        score += 2

    # --- Other important sections (1 point each) ---
    for sec in ["Projects", "Certifications", "Achievements"]:
        if sections[sec]["present"]:
            score += 1

    # --- Contact info (email or phone) ---
    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", lower):
        score += 1
    if re.search(r'(\+?\d[\d\-\s]{8,}\d)', lower):
        score += 1
    # --- Timeline (years) ---
    if re.search(r"\b\d{4}\b", lower):
        score += 1

    # --- Length ---
    if len(text.split()) > 250:
        score += 1

    # --- Bullet formatting ---
    total_bullets = sum(info["bullet_count"] for info in sections.values())
    if total_bullets >= 5:
        score += 1

    # --- Normalize and cap ---
    score_pct = int((score / total) * 100)
    return min(score_pct, 100)


# Skills Extraction Model - Dictionary Matching Approach

skills_df = pd.read_csv("skills.csv")
skills_list = [s.strip().lower() for s in skills_df["Skill"].dropna()]


# Sort skills by length (longest phrases first to avoid partial overlaps)
skills_list = sorted(skills_list, key=len, reverse=True)

def extract_skills_from_text(text):
    found_skills = set()
    # --- Dictionary matching ---
    text_lower = text.lower()
    for skill in skills_list:
        # Whole word / phrase match (case-insensitive)
        if re.search(rf"\b{re.escape(skill)}\b", text_lower):
            found_skills.add(skill)

    # --- Deduplicate while keeping order ---
    deduped = []
    for s in found_skills:
        if s not in deduped:
            deduped.append(s)

    return deduped


# --- Resume Summary, Suggestions, Role Prediction ---

def _fix_json_str(bad_json: str) -> str:
    """
    Attempt to fix common JSON issues:
    - Remove trailing commas before ] or }
    - Ensure quotes around keys
    """
    # Remove trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", bad_json)
    return fixed

def generate_summary_suggestions_and_role(resume_text: str):
    """
    Single LLM call that returns (summary, suggestions_list, job_role).
    Uses JSON output, with basic fixing if model output is malformed.
    """
    prompt = f"""
You are a professional resume analyst.

INSTRUCTIONS (follow exactly):
1) Write a single FORMAL PARAGRAPH summary of the resume. It should be atleast 200 words long.
2) Give exactly 10 concrete , actionable suggestions to improve this resume.
3) Predict the most likely job role.

OUTPUT (valid JSON only):
{{
  "summary": "<paragraph>",
  "suggestions": [
    "suggestion 1",
    "suggestion 2",
    "suggestion 3",
    "suggestion 4",
    "suggestion 5",
    "suggestion 6",
    "suggestion 7",
    "suggestion 8",
    "suggestion 9",
    "suggestion 10"
  ],
  "job_role": "<role>"
}}

Resume:
\"\"\"{resume_text}\"\"\"
"""

    resp = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.0,
    )

    raw = resp.choices[0].message.content.strip()

    try:
        obj = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise ValueError(f"Model did not return JSON:\n{raw}")
        fixed_json = _fix_json_str(match.group(0))
        obj = json.loads(fixed_json)

    summary = obj.get("summary", "").strip()
    suggestions = [s.strip() for s in obj.get("suggestions", [])]
    job_role = obj.get("job_role", "").strip()

    return summary, suggestions, job_role

# --- JD Ranking ---

# def rank_resumes(jd_text, resumes):

#     # 1) Extract JD skills deterministically 
#     jd_skills = extract_skills_from_text(jd_text)
#     jd_skills = [s.lower() for s in jd_skills]

#     # 2) Precompute JD embeddings 
#     jd_skill_embeddings = None
#     if jd_skills:
#         jd_skill_embeddings = embedding_model.encode(jd_skills, convert_to_tensor=True)

#     jd_text_embedding = embedding_model.encode(jd_text, convert_to_tensor=True)

#     ranked_list = []

#     for uploaded_file in resumes:
#         # Save to a temp file and ensure cleanup
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#             uploaded_file.save(tmp.name)
#             tmp_path = tmp.name

#         try:
#             text = extract_text_from_pdf(tmp_path)

#             # Extract resume skills deterministically
#             resume_skills = extract_skills_from_text(text)
#             resume_skills = [s.lower() for s in resume_skills]

#             # --- Skill match calculation ---
#             skill_match = 0.0
#             if resume_skills and jd_skill_embeddings is not None:
#                 resume_skill_embeddings = embedding_model.encode(resume_skills, convert_to_tensor=True)
#                 cosine_scores = util.cos_sim(resume_skill_embeddings, jd_skill_embeddings)  # shape R x J
#                 # For each JD skill (column) take best matching resume skill (max over rows), then average
#                 best_for_each_jd = cosine_scores.max(dim=0).values  # shape (J,)
#                 skill_match = float(best_for_each_jd.mean().item())

#             # --- Semantic similarity between whole resume and JD ---
#             semantic_score = float(util.cos_sim(
#                 embedding_model.encode(text, convert_to_tensor=True),
#                 jd_text_embedding
#             ).item())

#             # --- Clamp negative similarities to 0 and cap at 1 ---
#             skill_match = max(0.0, min(1.0, skill_match))
#             semantic_score = max(0.0, min(1.0, semantic_score))

#             # --- ATS score (0..1) ---
#             ats_score = calculate_ats_score(text) / 100.0
#             ats_score = max(0.0, min(1.0, ats_score))

#             # --- Final weighted score ---
#             final_score = 0.45 * skill_match + 0.35 * semantic_score + 0.2 * ats_score
#             final_score = max(0.0, min(1.0, final_score))

#             # Append rounded percentages
#             ranked_list.append({
#                 "filename": uploaded_file.filename,
#                 "skill_match": round(skill_match * 100, 1),
#                 "semantic_score": round(semantic_score * 100, 1),
#                 "ats_score": round(ats_score * 100, 1),
#                 "final_score": round(final_score * 100, 1)
#             })

#         finally:
#             try:
#                 os.remove(tmp_path)
#             except OSError:
#                 pass

#     # Sort by final_score desc
#     return sorted(ranked_list, key=lambda x: x['final_score'], reverse=True)


# --- Flask App ---
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/analyze_resume", methods=["GET", "POST"])
def analyze_resume():
    prediction = ats_score = section_stats = summary_text = skills = suggestions_list = None

    if request.method == "POST":
        file = request.files.get("resume")
        if file and file.filename.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            try:
                text = extract_text_from_pdf(tmp_path)
                skills = extract_skills_from_text(text)
                ats_score = calculate_ats_score(text)
                section_stats = analyze_sections(text)

                summary_text, suggestions_list, prediction = generate_summary_suggestions_and_role(text)

            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return render_template(
        "analyze_resume.html",
        prediction=prediction,
        ats_score=ats_score,
        sections=section_stats,
        summary=summary_text,      
        skills=skills,
        suggestions=suggestions_list   
    )


# @app.route("/jd_ranking", methods=["GET","POST"])
# def jd_ranking():
#     ranked_resumes = None
#     if request.method == "POST":
#         jd_text = request.form.get("jd_text")
#         uploaded_files = request.files.getlist("resumes")
#         if jd_text and uploaded_files:
#             ranked_resumes = rank_resumes(jd_text, uploaded_files)
#     return render_template("jd_ranking.html", ranked_resumes=ranked_resumes)

@app.route("/jd_ranking", methods=["GET"])
def jd_ranking():
    return "JD ranking is disabled in the public demo. Full version available locally."

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

