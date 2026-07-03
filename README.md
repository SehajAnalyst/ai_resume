# CareerFit AI 🎯

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fitcareer-ai.streamlit.app/)

### Resume Score Prediction & Skill Gap Analysis System

A machine-learning powered resume evaluation tool built with Python, Streamlit, and scikit-learn.

A machine-learning powered resume evaluation tool built with Python, Streamlit, and scikit-learn.
CareerFit AI is a machine learning-based web application that helps users evaluate their resumes by analyzing skills, identifying missing competencies, and providing personalized improvement suggestions.

The goal of this project is to give job seekers a better understanding of how well their resumes align with different career roles and what areas they can improve.

---

## Features

- Upload a PDF resume and get instant analysis.
- Extract and identify 40+ technical and soft skills.
- Detect skill gaps for different job roles.
- Generate a resume score out of 100 based on multiple factors.
- Predict resume categories using a Random Forest model.
- Provide personalized recommendations for improvement.
- Display interactive charts and visual insights.

---

## Setup and Installation

### Install dependencies

```bash
pip install -r requirements.txt
```

### (Optional) Train the machine learning model

```bash
python train_model.py
```

> The model automatically trains itself on the first run if a saved model is not available.

### Run the application

```bash
streamlit run app.py
```

---

## Project Structure

```text
careerfit_ai/
├── app.py
├── core.py
├── pdf_extractor.py
├── visualizations.py
├── train_model.py
├── requirements.txt
├── rf_model.pkl
├── label_encoder.pkl
├── synthetic_resume_dataset.csv
└── PROJECT_REPORT.md
```

---

## Resume Scoring System

| Component | Maximum Score |
|-----------|---------------|
| Skill Match | 40 |
| Projects | 20 |
| Experience | 15 |
| Education | 10 |
| Keywords | 10 |
| Formatting | 5 |
| **Total** | **100** |

### Score Categories

| Score Range | Category |
|-------------|----------|
| 0 – 40 | Weak |
| 41 – 70 | Average |
| 71 – 85 | Good |
| 86 – 100 | Excellent |

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

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly
- pdfplumber
- Regular Expressions (Regex)

---

## What I Learned

Building this project helped me gain hands-on experience in:

- Machine Learning model development
- Feature engineering and resume scoring techniques
- PDF text extraction and preprocessing
- Building interactive web applications using Streamlit
- Data visualization and user-centric design

---

## Future Improvements

- Resume and Job Description matching
- ATS score prediction
- AI-generated resume recommendations
- Interview question generation
- Multi-language resume support

---

## Author

**Sehaj Oberoi**

GitHub: https://github.com/SehajAnalyst
