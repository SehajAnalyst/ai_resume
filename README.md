# CareerFit AI 🎯
### Resume Score Prediction & Skill Gap Analysis System

A machine-learning powered resume evaluation tool built with Python, Streamlit, and scikit-learn.

---

## Features
- Upload PDF resume → instant analysis
- Skill extraction from resume text (40+ skills)
- Role-specific skill gap detection (7 job roles)
- Multi-component resume scoring (out of 100)
- Random Forest ML model predicts resume category
- Personalized improvement suggestions
- Interactive charts: gauge, bar breakdown, pie chart

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Pre-train the ML model
```bash
python train_model.py
```
> The model auto-trains on first Streamlit run if not pre-trained.

### 3. Run the application
```bash
streamlit run app.py
```

---

## Project Structure
```
careerfit_ai/
├── app.py                      # Streamlit web application
├── core.py                     # Skills, scoring logic, ML model
├── pdf_extractor.py            # PDF text extraction
├── visualizations.py           # Plotly charts
├── train_model.py              # Standalone model training
├── requirements.txt            # Dependencies
├── rf_model.pkl                # Saved ML model (after training)
├── label_encoder.pkl           # Saved label encoder (after training)
├── synthetic_resume_dataset.csv # Generated training data
└── PROJECT_REPORT.md           # Full project report
```

---

## Scoring System
| Component    | Max Score |
|-------------|-----------|
| Skill Match | 40        |
| Projects    | 20        |
| Experience  | 15        |
| Education   | 10        |
| Keywords    | 10        |
| Formatting  | 5         |
| **Total**   | **100**   |

## Categories
| Score | Category  |
|-------|-----------|
| 0–40  | Weak      |
| 41–70 | Average   |
| 71–85 | Good      |
| 86–100| Excellent |

---

## Supported Job Roles
- Data Analyst
- Data Scientist
- Machine Learning Engineer
- Software Developer
- Web Developer
- Business Analyst
- Cloud Engineer

---

## Technologies
Python · Streamlit · scikit-learn · Pandas · NumPy · Plotly · pdfplumber · Regex

---

Built as a Final Year Machine Learning Project.
