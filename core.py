"""
CareerFit AI - Core Logic Module
Handles: skill dictionaries, role requirements, scoring, ML model training
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 1. SKILL DICTIONARY
# ─────────────────────────────────────────────

ALL_SKILLS = [
    # Programming Languages
    "python",
    "java",
    "c++",
    "c",
    "r",
    "scala",
    "kotlin",
    "swift",
    # Data / Analytics
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "excel",
    "power bi",
    "tableau",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "plotly",
    # Machine Learning / AI
    "machine learning",
    "deep learning",
    "scikit-learn",
    "tensorflow",
    "keras",
    "pytorch",
    "nlp",
    "computer vision",
    "data cleaning",
    "data visualization",
    "eda",
    "statistics",
    "a/b testing",
    "feature engineering",
    "model deployment",
    # Web Development
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "django",
    "flask",
    "angular",
    "vue",
    "bootstrap",
    "rest api",
    "graphql",
    # Cloud / DevOps
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "github",
    "linux",
    "ci/cd",
    "terraform",
    # Soft Skills
    "communication",
    "leadership",
    "problem solving",
    "teamwork",
    "critical thinking",
    "time management",
    "project management",
    # Business
    "business analysis",
    "agile",
    "scrum",
    "jira",
    "power apps",
]

# ─────────────────────────────────────────────
# 2. ROLE-SPECIFIC REQUIRED SKILLS
# ─────────────────────────────────────────────

ROLE_REQUIREMENTS = {
    "Data Analyst": {
        "required": [
            "sql",
            "excel",
            "python",
            "power bi",
            "tableau",
            "pandas",
            "numpy",
            "data visualization",
            "eda",
            "statistics",
            "data cleaning",
            "matplotlib",
            "seaborn",
        ],
        "keywords": [
            "dashboard",
            "report",
            "insight",
            "analysis",
            "kpi",
            "visualization",
            "trend",
            "metric",
            "query",
        ],
    },
    "Data Scientist": {
        "required": [
            "python",
            "machine learning",
            "deep learning",
            "statistics",
            "pandas",
            "numpy",
            "scikit-learn",
            "tensorflow",
            "feature engineering",
            "eda",
            "sql",
            "data cleaning",
            "nlp",
            "a/b testing",
        ],
        "keywords": [
            "model",
            "prediction",
            "classification",
            "regression",
            "clustering",
            "accuracy",
            "training",
            "dataset",
            "experiment",
        ],
    },
    "Machine Learning Engineer": {
        "required": [
            "python",
            "machine learning",
            "deep learning",
            "tensorflow",
            "pytorch",
            "scikit-learn",
            "model deployment",
            "docker",
            "flask",
            "rest api",
            "git",
            "feature engineering",
            "nlp",
        ],
        "keywords": [
            "deploy",
            "pipeline",
            "inference",
            "production",
            "api",
            "training",
            "optimization",
            "mlops",
        ],
    },
    "Software Developer": {
        "required": [
            "python",
            "java",
            "c++",
            "git",
            "github",
            "sql",
            "data structures",
            "algorithms",
            "rest api",
            "linux",
        ],
        "keywords": [
            "development",
            "software",
            "application",
            "coding",
            "debugging",
            "unit test",
            "version control",
            "agile",
        ],
    },
    "Web Developer": {
        "required": [
            "html",
            "css",
            "javascript",
            "react",
            "node.js",
            "git",
            "rest api",
            "mongodb",
            "bootstrap",
            "angular",
        ],
        "keywords": [
            "frontend",
            "backend",
            "responsive",
            "ui",
            "ux",
            "web app",
            "deployment",
            "dom",
            "api",
        ],
    },
    "Business Analyst": {
        "required": [
            "excel",
            "sql",
            "power bi",
            "tableau",
            "business analysis",
            "agile",
            "scrum",
            "jira",
            "communication",
            "statistics",
        ],
        "keywords": [
            "requirement",
            "stakeholder",
            "process",
            "workflow",
            "documentation",
            "brd",
            "frd",
            "user story",
            "sprint",
        ],
    },
    "Cloud Engineer": {
        "required": [
            "aws",
            "azure",
            "gcp",
            "docker",
            "kubernetes",
            "terraform",
            "linux",
            "git",
            "ci/cd",
            "python",
            "rest api",
        ],
        "keywords": [
            "cloud",
            "infrastructure",
            "deployment",
            "scalability",
            "devops",
            "container",
            "serverless",
            "monitoring",
        ],
    },
}

# ─────────────────────────────────────────────
# 3. TEXT PREPROCESSING
# ─────────────────────────────────────────────


def preprocess_text(text: str) -> str:
    """Convert to lowercase, remove symbols, clean extra spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)  # keep + # . for C++, C#, etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# 4. SKILL EXTRACTION
# ─────────────────────────────────────────────


def extract_skills(text: str) -> list:
    """Find all matching skills from the ALL_SKILLS list in the resume text."""
    cleaned = preprocess_text(text)
    found = []

    for skill in ALL_SKILLS:
        # Use word-boundary match so "r" doesn't match inside "director"
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, cleaned):
            found.append(skill)

    return found


# ─────────────────────────────────────────────
# 5. PROJECT SECTION EXTRACTION & STRENGTH
# ─────────────────────────────────────────────


def extract_projects_section(resume_text: str) -> str:
    """Extract the Projects section from a resume.

    Supports headings like:
    - Projects
    - Project
    - Academic Projects
    - Personal Projects
    - Key Projects
    """

    if not resume_text:
        return ""

    lines = resume_text.split("\n")

    headings = [
        "projects",
        "project",
        "academic projects",
        "personal projects",
        "key projects",
    ]

    start = None
    end = len(lines)

    for i, line in enumerate(lines):
        if line.strip().lower() in headings:
            start = i + 1
            break

    if start is None:
        return ""

    # Find next section heading (heuristic: uppercase short line)
    for i in range(start, len(lines)):
        line = lines[i].strip()
        if (
            line.isupper()
            and len(line.split()) <= 4
            and line.lower() not in headings
        ):
            end = i
            break

    return "\n".join(lines[start:end]).strip()


def calculate_project_strength(resume_text: str, job_description: str) -> int:
    """Calculate project strength as 0..100 percentage.

    - Extracts the projects section from the resume.
    - Compares it against the job description using TF-IDF + cosine similarity.
    """

    projects_section = extract_projects_section(resume_text or "")
    if not projects_section.strip():
        return 0

    job_description = job_description or ""

    # If JD is empty, strength should be 0
    if not job_description.strip():
        return 0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )

    tfidf = vectorizer.fit_transform([projects_section, job_description])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    # similarity is in [0,1] for cosine similarity of tf-idf vectors
    similarity = float(max(0.0, min(1.0, similarity)))
    return int(round(similarity * 100))


# ─────────────────────────────────────────────
# 6. EXPERIENCE / EDUCATION / KEYWORDS / FORMATTING
# ─────────────────────────────────────────────


def detect_experience(text: str) -> float:
    """Return an experience score 0-15."""
    cleaned = (text or "").lower()
    score = 0

    if re.search(r"\b(internship|intern)\b", cleaned):
        score += 7
    if re.search(r"\b(experience|worked at|employed)\b", cleaned):
        score += 5

    years = re.findall(r"(\d+)\s*\+?\s*year", cleaned)
    if years:
        total_years = sum(int(y) for y in years)
        score += min(total_years * 2, 8)

    return min(score, 15)


def detect_education(text: str) -> int:
    """Return education score 0-10."""
    cleaned = (text or "").lower()
    if re.search(r"\b(phd|doctorate)\b", cleaned):
        return 10
    if re.search(r"\b(master|mtech|msc|mba|m\.tech)\b", cleaned):
        return 9
    if re.search(r"\b(bachelor|btech|bsc|b\.tech|be|b\.e)\b", cleaned):
        return 7
    if re.search(r"\b(diploma|12th|higher secondary)\b", cleaned):
        return 4
    return 3


def detect_keywords(text: str, role: str) -> int:
    """Return keyword match score 0-10 based on role-specific keywords."""
    cleaned = preprocess_text(text or "")
    keywords = ROLE_REQUIREMENTS[role]["keywords"]

    matched = sum(
        1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", cleaned)
    )
    ratio = matched / len(keywords) if keywords else 0
    return round(ratio * 10)


def detect_formatting(text: str) -> int:
    """Return formatting score 0-5."""
    cleaned = (text or "").lower()
    score = 0

    sections = ["education", "experience", "skills", "project", "summary", "objective"]
    found_sections = sum(1 for s in sections if s in cleaned)
    score += min(found_sections, 3)

    if re.search(r"[\w.-]+@[\w.-]+", text or ""):
        score += 1
    if re.search(r"\b\d{10}\b", text or "") or re.search(r"\+\d[\d\s-]{8,}", text or ""):
        score += 1

    return min(score, 5)


# ─────────────────────────────────────────────
# 6. SCORE CALCULATION
# ─────────────────────────────────────────────


def calculate_score(resume_text: str, role: str) -> dict:
    """Master scoring function (role-based)."""

    required_skills = ROLE_REQUIREMENTS[role]["required"]

    found_skills = extract_skills(resume_text)
    matched = [s for s in found_skills if s in required_skills]
    missing = [s for s in required_skills if s not in found_skills]

    skill_match_pct = (
        round(len(matched) / len(required_skills) * 100, 1)
        if required_skills
        else 0
    )

    skill_score = round(skill_match_pct / 100 * 40)

    # Role-based project strength: approximate JD using role keywords
    # (We still compare projects section against a text proxy of the role.)
    role_jd_proxy = " ".join(ROLE_REQUIREMENTS[role].get("keywords", []))
    project_strength = calculate_project_strength(resume_text, role_jd_proxy)
    project_score = round(project_strength / 100 * 20)

    experience_score = detect_experience(resume_text)
    education_score = detect_education(resume_text)
    keyword_score = detect_keywords(resume_text, role)
    formatting_score = detect_formatting(resume_text)

    final_score = (
        skill_score
        + project_score
        + experience_score
        + education_score
        + keyword_score
        + formatting_score
    )
    final_score = min(final_score, 100)

    if final_score <= 40:
        category = "Weak"
    elif final_score <= 70:
        category = "Average"
    elif final_score <= 85:
        category = "Good"
    else:
        category = "Excellent"

    return {
        "role": role,
        "found_skills": found_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        "total_skills_found": len(found_skills),
        "matched_skills_count": len(matched),
        "missing_skills_count": len(missing),
        "skill_match_percentage": skill_match_pct,
        "skill_score": skill_score,
        "project_strength": project_strength,
        "project_score": project_score,
        "experience_score": experience_score,
        "education_score": education_score,
        "keyword_score": keyword_score,
        "formatting_score": formatting_score,
        "final_score": final_score,
        "category": category,
    }


# ─────────────────────────────────────────────
# 6B. JOB DESCRIPTION MATCHING
# ─────────────────────────────────────────────


def extract_required_skills_from_jd(job_description: str) -> list:
    return extract_skills(job_description)


def calculate_jd_match(resume_text: str, job_description: str) -> dict:
    resume_skills = extract_skills(resume_text)
    jd_required_skills = extract_required_skills_from_jd(job_description)

    matched_skills = [skill for skill in resume_skills if skill in jd_required_skills]
    missing_skills = [skill for skill in jd_required_skills if skill not in resume_skills]

    if jd_required_skills:
        skill_match_percentage = round(len(matched_skills) / len(jd_required_skills) * 100, 1)
    else:
        skill_match_percentage = 0

    # Text similarity (full resume vs JD)
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=5000
    )
    vectors = vectorizer.fit_transform([resume_text, job_description])
    text_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    text_similarity_percentage = round(float(text_similarity) * 100, 1)

    skill_score = round(skill_match_percentage / 100 * 40)

    project_strength = calculate_project_strength(resume_text, job_description)
    project_score = round(project_strength / 100 * 20)

    experience_score = detect_experience(resume_text)
    education_score = detect_education(resume_text)
    keyword_score = round(text_similarity_percentage / 100 * 10)
    formatting_score = detect_formatting(resume_text)

    final_score = (
        skill_score
        + project_score
        + experience_score
        + education_score
        + keyword_score
        + formatting_score
    )
    final_score = min(final_score, 100)

    if final_score <= 40:
        category = "Weak"
    elif final_score <= 70:
        category = "Average"
    elif final_score <= 85:
        category = "Good"
    else:
        category = "Excellent"

    return {
        "role": "Custom Job Description",
        "required_skills": jd_required_skills,
        "found_skills": resume_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "total_skills_found": len(resume_skills),
        "matched_skills_count": len(matched_skills),
        "missing_skills_count": len(missing_skills),
        "skill_match_percentage": skill_match_percentage,
        "text_similarity_percentage": text_similarity_percentage,
        "skill_score": skill_score,
        "project_strength": project_strength,
        "project_score": project_score,
        "experience_score": experience_score,
        "education_score": education_score,
        "keyword_score": keyword_score,
        "formatting_score": formatting_score,
        "final_score": final_score,
        "category": category,
    }


def generate_jd_suggestions(result: dict) -> list:
    suggestions = []

    if result["skill_match_percentage"] < 50:
        suggestions.append(
            f"Your resume matches only {result['skill_match_percentage']}% of the required JD skills. "
            "Add the most important missing skills if you genuinely know them."
        )

    if result["missing_skills"]:
        top_missing = result["missing_skills"][:7]
        suggestions.append(
            "Missing JD skills: "
            + ", ".join(top_missing)
            + ". Add these to your Skills or Projects section only if you have actually used them."
        )

    if result["text_similarity_percentage"] < 35:
        suggestions.append(
            f"Your resume text similarity with the JD is only {result['text_similarity_percentage']}%. "
            "Use more job-description-related terms naturally in your project and experience bullet points."
        )

    # Project suggestions based on project_strength
    if result.get("project_strength", 0) < 40:
        suggestions.append(
            "Improve your projects: add 1-3 strong projects (or bullets) that directly use the job’s core technologies."
        )
    elif result.get("project_strength", 0) < 65:
        suggestions.append(
            "You have some project relevance, but strengthen it further by aligning project outcomes/tech stack to the JD."
        )

    if result["experience_score"] < 7:
        suggestions.append(
            "Internship or work experience is not clearly detected. Mention internships, freelance work, or real project experience clearly."
        )

    suggestions.append(
        "Add measurable achievements. Example: 'Improved model accuracy by 12%' or 'Built dashboard reducing manual reporting time by 30%'."
    )

    return suggestions


# ─────────────────────────────────────────────
# 7. SUGGESTIONS ENGINE (role-based)
# ─────────────────────────────────────────────


def generate_suggestions(result: dict) -> list:
    suggestions = []

    if result["skill_match_percentage"] < 50:
        suggestions.append(
            f"Your skill match is only {result['skill_match_percentage']}%. "
            f"Focus on learning the missing skills for {result['role']}."
        )

    if result["missing_skills"]:
        top_missing = result["missing_skills"][:5]
        suggestions.append(
            f"Add these missing skills: {', '.join(top_missing)}. "
            "Use online courses (Coursera, Udemy) or build projects using them."
        )

    # Project suggestions based on project_strength
    if result.get("project_strength", 0) < 40:
        suggestions.append(
            "Improve your projects by adding 2-3 clear project bullets that show relevant technologies and measurable results."
        )

    if result["experience_score"] < 7:
        suggestions.append(
            "No internship or work experience detected. Apply for internships or add freelance/open-source contributions."
        )

    if result["keyword_score"] < 5:
        suggestions.append(
            f"Your resume lacks important {result['role']} keywords. "
            "Tailor your resume content to match the job description language."
        )

    if result["formatting_score"] < 3:
        suggestions.append(
            "Resume structure is weak. Include clear sections: Summary, Skills, Education, Experience, Projects. Add your email and phone number."
        )

    if result["education_score"] < 7:
        suggestions.append("Mention your degree clearly (B.Tech, B.Sc, etc.) with year and institution.")

    suggestions.append(
        "Add measurable achievements wherever possible. Example: 'Improved model accuracy by 15%' or 'Reduced report time by 30%'."
    )

    suggestions.append(
        "Include links to your GitHub profile or portfolio. Recruiters check this to verify your technical skills."
    )

    return suggestions


# ─────────────────────────────────────────────
# 8. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────


def generate_synthetic_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    total_skills_found = rng.integers(2, 25, n)
    matched_skills_count = np.array([rng.integers(0, s + 1) for s in total_skills_found])
    missing_skills_count = rng.integers(0, 15, n)
    skill_match_pct = np.clip(
        matched_skills_count / np.maximum(total_skills_found, 1) * 100, 0, 100
    )

    # replace project_count with project_strength
    project_strength = rng.integers(0, 101, n)

    experience_score = rng.integers(0, 16, n).astype(float)
    education_score = rng.choice([3, 4, 7, 9, 10], n)
    keyword_match_score = rng.integers(0, 11, n)
    formatting_score = rng.integers(0, 6, n)

    skill_score = np.round(skill_match_pct / 100 * 40).astype(int)
    project_score = np.round(project_strength / 100 * 20).astype(int)

    final_score = np.clip(
        skill_score
        + project_score
        + experience_score
        + education_score
        + keyword_match_score
        + formatting_score,
        0,
        100,
    ).astype(int)

    def to_category(s: int) -> str:
        if s <= 40:
            return "Weak"
        if s <= 70:
            return "Average"
        if s <= 85:
            return "Good"
        return "Excellent"

    category = np.array([to_category(s) for s in final_score])

    return pd.DataFrame(
        {
            "total_skills_found": total_skills_found,
            "matched_skills_count": matched_skills_count,
            "missing_skills_count": missing_skills_count,
            "skill_match_percentage": np.round(skill_match_pct, 1),
            "project_strength": project_strength,
            "experience_score": experience_score,
            "education_score": education_score,
            "keyword_match_score": keyword_match_score,
            "formatting_score": formatting_score,
            "final_score": final_score,
            "category": category,
        }
    )


# ─────────────────────────────────────────────
# 9. ML MODEL TRAINING & PERSISTENCE
# ─────────────────────────────────────────────

FEATURES = [
    "total_skills_found",
    "matched_skills_count",
    "missing_skills_count",
    "skill_match_percentage",
    "project_strength",
    "experience_score",
    "education_score",
    "keyword_match_score",
    "formatting_score",
    "final_score",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")


def train_model(df: pd.DataFrame = None) -> tuple:
    if df is None:
        df = generate_synthetic_dataset()

    X = df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df["category"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    return model, le, round(acc * 100, 2)


def load_or_train_model() -> tuple:
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(ENCODER_PATH, "rb") as f:
                le = pickle.load(f)
            return model, le, None
        except Exception:
            # feature mismatch or pickle incompatibility -> retrain
            return train_model()
    return train_model()


def predict_category(result: dict, model, le) -> str:
    row = pd.DataFrame(
        [
            {
                "total_skills_found": result["total_skills_found"],
                "matched_skills_count": result["matched_skills_count"],
                "missing_skills_count": result["missing_skills_count"],
                "skill_match_percentage": result["skill_match_percentage"],
                "project_strength": result["project_strength"],
                "experience_score": result["experience_score"],
                "education_score": result["education_score"],
                "keyword_match_score": result["keyword_score"],
                "formatting_score": result["formatting_score"],
                "final_score": result["final_score"],
            }
        ]
    )
    pred = model.predict(row)
    return le.inverse_transform(pred)[0]

