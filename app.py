"""
CareerFit AI - Main Streamlit Application
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import base64

# ── Local modules ──────────────────────────────────────────────────────────────
from language_analyzer import analyze_language_quality
from core import (
    calculate_jd_match,
    generate_jd_suggestions,
    load_or_train_model,
    predict_category,
)

from pdf_extractor import extract_text_from_pdf
from visualizations import score_breakdown_chart, skill_pie_chart, score_gauge


def show_pdf_preview(uploaded_file):
    """
    Display uploaded PDF inside Streamlit using iframe.
    """
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="700px" 
        type="application/pdf">
    </iframe>
    """

    st.markdown(pdf_display, unsafe_allow_html=True)
    uploaded_file.seek(0)


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CareerFit AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F9FAFB; }

    /* Score badges */
    .badge-weak      { background:#FF5252; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .badge-average   { background:#FFA726; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .badge-good      { background:#66BB6A; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .badge-excellent { background:#1E88E5; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }

    /* Skill chips */
    .chip-green { display:inline-block; background:#E8F5E9; color:#2E7D32;
                  border:1px solid #A5D6A7; border-radius:16px;
                  padding:2px 10px; margin:3px; font-size:13px; }
    .chip-red   { display:inline-block; background:#FFEBEE; color:#C62828;
                  border:1px solid #EF9A9A; border-radius:16px;
                  padding:2px 10px; margin:3px; font-size:13px; }

    /* Section header */
    .section-header { font-size:1.1rem; font-weight:700;
                      border-left:4px solid #1E88E5;
                      padding-left:10px; margin-top:20px; margin-bottom:8px; }

    /* Suggestion items */
    .suggestion-box { background:#EFF6FF; border-left:3px solid #1E88E5;
                       padding:10px 14px; border-radius:6px; margin:6px 0; font-size:14px;
                       color:#1a1a2e !important; }

    /* Language quality cards */
    .lq-card { border-radius:10px; padding:14px 16px; margin:10px 0;
               border:1px solid #e0e0e0; background:#fff; color:#1a1a2e !important; }
    .lq-tag-ai     { display:inline-block; background:#FFF3E0; color:#E65100 !important;
                     border:1px solid #FFCC80; border-radius:12px;
                     padding:2px 10px; font-size:12px; font-weight:600; margin-bottom:8px; }
    .lq-tag-weak   { display:inline-block; background:#FFF9C4; color:#827717 !important;
                     border:1px solid #F9A825; border-radius:12px;
                     padding:2px 10px; font-size:12px; font-weight:600; margin-bottom:8px; }
    .lq-tag-bad    { display:inline-block; background:#FCE4EC; color:#880E4F !important;
                     border:1px solid #F48FB1; border-radius:12px;
                     padding:2px 10px; font-size:12px; font-weight:600; margin-bottom:8px; }
    .lq-original   { background:#FFEBEE; border-radius:6px; padding:8px 12px;
                     font-family:monospace; font-size:13px; color:#B71C1C !important;
                     text-decoration:line-through; margin:4px 0; }
    .lq-rewritten  { background:#E8F5E9; border-radius:6px; padding:8px 12px;
                     font-family:monospace; font-size:13px; color:#1B5E20 !important; margin:4px 0; }
    .lq-problem    { font-size:13px; color:#444 !important; margin:6px 0 4px 0; font-style:italic; }
    .lq-arrow      { text-align:center; font-size:18px; color:#9E9E9E !important; margin:2px 0; }
    .human-score-bar { height:16px; border-radius:8px;
                       background:linear-gradient(90deg,#FF5252,#FFA726,#66BB6A); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ML MODEL
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Training ML model… (first run only)")
def get_model():
    model, le, acc = load_or_train_model()
    return model, le, acc


model, le, model_acc = get_model()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/resume.png", width=80)
    st.title("CareerFit AI")
    st.markdown("**Resume Score Prediction & Skill Gap Analysis**")
    st.divider()

    st.markdown("#### 📤 Upload Your Resume")
    uploaded_file = st.file_uploader(
        "PDF format only",
        type=["pdf"],
        label_visibility="collapsed",
    )

    st.markdown("#### 🧾 Paste Job Description")
    job_description = st.text_area(
        "Job Description",
        height=220,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
    )

    st.divider()
    analyze_btn = st.button(
        "🔍 Analyze Resume",
        use_container_width=True,
        type="primary",
    )

    if model_acc:
        st.caption(f"✅ ML Model Accuracy: **{model_acc}%**")

    st.divider()
    st.markdown("""
    **How it works:**
    1. Upload your resume (PDF)
    2. Paste the job description
    3. Click **Analyze Resume**
    4. View score, gaps & suggestions
    """)

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

st.title("🎯 CareerFit AI — Resume Analyzer")
st.markdown("Powered by NLP + Random Forest ML · Instant skill gap detection & personalized advice")
st.divider()

# ── Demo mode note ─────────────────────────────────────────────────────────────

if not uploaded_file:
    st.info(
        "👈 Upload your resume PDF from the sidebar and paste a job description to check resume-JD match.",
        icon="ℹ️",
    )
    st.stop()

# ── RUN ANALYSIS ───────────────────────────────────────────────────────────────

if uploaded_file is not None and job_description.strip():

    with st.expander("📄 Preview Uploaded Resume PDF", expanded=False):
        show_pdf_preview(uploaded_file)

    # Extract PDF text
    with st.spinner("Extracting text from PDF…"):
        try:
            resume_text = extract_text_from_pdf(uploaded_file)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if len(resume_text.strip()) < 50:
        st.error(
            "The extracted text is too short. "
            "Make sure the PDF is not image-based / scanned."
        )
        st.stop()

    # Run scoring
    with st.spinner("Analyzing your resume…"):
        result = calculate_jd_match(resume_text, job_description)
        ml_category = predict_category(result, model, le)
        suggestions = generate_jd_suggestions(result)

else:
    st.info("Upload your resume PDF and paste the job description to start analysis.")
    st.stop()

# ── TOP SUMMARY ROW ─────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

badge_class = f"badge-{result['category'].lower()}"

col1.metric("🏆 Resume-JD Score", f"{result['final_score']} / 100")
col2.metric("🎯 Skill Match", f"{result['skill_match_percentage']}%")
col3.metric("📄 JD Similarity", f"{result['text_similarity_percentage']}%")
col4.metric("📁 Project Strength", f"{result['project_strength']}%")

st.markdown(
    f"**Category:** <span class='{badge_class}'>{result['category']}</span> &nbsp;|&nbsp; "
    f"**Analysis Type:** Resume vs Job Description Match",
    unsafe_allow_html=True,
)

st.divider()

# ── CHARTS ROW ──────────────────────────────────────────────────────────────

c1, c2, c3 = st.columns([2, 1.5, 1.5])

with c1:
    st.plotly_chart(score_breakdown_chart(result), use_container_width=True)

with c2:
    st.plotly_chart(skill_pie_chart(result), use_container_width=True)

with c3:
    st.plotly_chart(
        score_gauge(result["final_score"], result["category"]),
        use_container_width=True,
    )

st.divider()

# ── SKILLS SECTION ──────────────────────────────────────────────────────────

s1, s2 = st.columns(2)

with s1:
    st.markdown(
        '<div class="section-header">✅ Matched Skills</div>',
        unsafe_allow_html=True,
    )

    if result["matched_skills"]:
        chips = " ".join([
            f'<span class="chip-green">{s}</span>'
            for s in sorted(result["matched_skills"])
        ])
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.warning("No matching skills found for this role.")

with s2:
    st.markdown(
        '<div class="section-header">❌ Missing Skills</div>',
        unsafe_allow_html=True,
    )

    if result["missing_skills"]:
        chips = " ".join([
            f'<span class="chip-red">{s}</span>'
            for s in sorted(result["missing_skills"])
        ])
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.success("Great! You have all required skills for this role.")

st.divider()

# ── ALL EXTRACTED SKILLS ────────────────────────────────────────────────────

with st.expander("🔍 All Skills Extracted from Your Resume"):
    if result["found_skills"]:
        chips = " ".join([
            f'<span class="chip-green">{s}</span>'
            for s in sorted(result["found_skills"])
        ])
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.warning("No skills detected. Check if your resume text is properly formatted.")

# ── DETAILED SCORE TABLE ────────────────────────────────────────────────────

with st.expander("📋 Detailed Score Breakdown"):
    score_df = pd.DataFrame({
        "Component": [
            "Skill Match",
            "Projects",
            "Experience",
            "Education",
            "Keywords",
            "Formatting",
            "TOTAL",
        ],
        "Your Score": [
            result["skill_score"],
            result["project_score"],
            result["experience_score"],
            result["education_score"],
            result["keyword_score"],
            result["formatting_score"],
            result["final_score"],
        ],
        "Max Score": [40, 20, 15, 10, 10, 5, 100],
    })

    score_df["Percentage"] = (
        score_df["Your Score"] / score_df["Max Score"] * 100
    ).round(1).astype(str) + "%"

    st.dataframe(score_df, use_container_width=True, hide_index=True)

st.divider()

# ── SUGGESTIONS ─────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">💡 Personalized Improvement Suggestions</div>',
    unsafe_allow_html=True,
)

for i, tip in enumerate(suggestions, 1):
    st.markdown(
        f'<div class="suggestion-box"><b>{i}.</b> {tip}</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── LANGUAGE QUALITY ANALYZER ───────────────────────────────────────────────

st.markdown(
    '<div class="section-header">🧠 Language Quality Analyzer</div>',
    unsafe_allow_html=True,
)

# Show key status
from language_analyzer import _get_groq_key

groq_key_set = bool(_get_groq_key())

if groq_key_set:
    st.info(
        "✅ Groq API key found. It will be verified when you run language analysis.",
        icon="🤖",
    )
else:
    st.error(
        "❌ Groq API key not found. "
        "Open `.streamlit/secrets.toml`, replace the placeholder with your real key, "
        "then restart the app. Get your free key at https://console.groq.com"
    )

st.markdown(
    "Analyzes your resume using Groq AI — detects weak language, buzzwords, "
    "and informal phrases with specific rewrites."
)

if st.button("🔎 Run Language Analysis", use_container_width=False, disabled=not groq_key_set):
    with st.spinner("Analyzing with Groq AI (Llama 3)… please wait"):
        try:
            lq = analyze_language_quality(resume_text)
            st.session_state["lq_result"] = lq
        except ValueError as e:
            st.error(str(e))
            st.session_state["lq_result"] = None
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.session_state["lq_result"] = None

# Display cached result if available
lq = st.session_state.get("lq_result")

if lq:
    # ── Mode badge ─────────────────────────────────────────────────────────
    st.markdown(
        '<span style="background:#1E88E5;color:white;padding:4px 12px;border-radius:20px;'
        'font-size:13px;font-weight:600;">🤖 Analyzed by Groq AI — Llama 3</span>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Human Score Meter ──────────────────────────────────────────────────
    hs = lq.get("human_score", 50)
    hs_color = "#FF5252" if hs < 40 else "#FFA726" if hs < 70 else "#4CAF50"
    hs_label = "Robotic" if hs < 40 else "Mixed" if hs < 70 else "Human"

    col_a, col_b = st.columns([3, 1])

    with col_a:
        st.markdown(
            f"""
            <div style="margin:12px 0 4px 0; font-size:13px; color:#555;">
              Human-sounding score &nbsp;
              <b style="color:{hs_color}">{hs}/100 — {hs_label}</b>
            </div>
            <div style="background:#e0e0e0; border-radius:8px; height:16px; overflow:hidden;">
              <div style="width:{hs}%; background:{hs_color}; height:16px; border-radius:8px;
                          transition:width 0.6s ease;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        issues_count = len(lq.get("issues", []))
        pc = lq.get("priority_counts", {})
        st.metric("Issues Found", f"{issues_count}  ({pc.get('high', 0)} critical)")

    # ── Summary ────────────────────────────────────────────────────────────
    st.info(lq.get("summary", ""), icon="💬")

    issues = lq.get("issues", [])

    if not issues:
        st.success("No significant language issues detected. Your resume reads naturally and professionally.")
    else:
        # ── Legend ─────────────────────────────────────────────────────────
        type_counts = {}

        for issue in issues:
            issue_type = issue.get("type", "weak_language")
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1

        legend_parts = []

        if type_counts.get("ai_sounding"):
            legend_parts.append(
                f'<span class="lq-tag-ai">🤖 AI/Buzzword: {type_counts["ai_sounding"]}</span>'
            )

        if type_counts.get("weak_language"):
            legend_parts.append(
                f'<span class="lq-tag-weak">⚠️ Weak language: {type_counts["weak_language"]}</span>'
            )

        if type_counts.get("unprofessional"):
            legend_parts.append(
                f'<span class="lq-tag-bad">🚫 Unprofessional: {type_counts["unprofessional"]}</span>'
            )

        st.markdown("&nbsp;&nbsp;".join(legend_parts), unsafe_allow_html=True)
        st.markdown("---")

        # ── Issue cards ─────────────────────────────────────────────────────
        for i, issue in enumerate(issues, 1):
            itype = issue.get("type", "weak_language")
            priority = issue.get("priority", "medium")
            original = issue.get("original", "")
            problem = issue.get("problem", "")
            rewrite = issue.get("rewritten", "")

            tag_class = {
                "ai_sounding": "lq-tag-ai",
                "weak_language": "lq-tag-weak",
                "unprofessional": "lq-tag-bad",
            }.get(itype, "lq-tag-weak")

            tag_label = {
                "ai_sounding": "🤖 AI/Buzzword",
                "weak_language": "⚠️ Weak language",
                "unprofessional": "🚫 Unprofessional",
            }.get(itype, "⚠️ Weak language")

            priority_badge_color = {
                "high": "#FF5252",
                "medium": "#FFA726",
                "low": "#66BB6A",
            }.get(priority, "#FFA726")

            priority_label = priority.upper()
            rewrite_html = rewrite.replace("\n", "<br>")

            st.markdown(
                f"""
                <div class="lq-card">
                  <span class="{tag_class}">{tag_label}</span>
                  &nbsp;
                  <span style="background:{priority_badge_color};color:white;padding:2px 9px;
                               border-radius:12px;font-size:11px;font-weight:700;">
                    {priority_label} PRIORITY
                  </span>
                  &nbsp;<span style="font-size:12px;color:#999;">#{i}</span>
                  <div class="lq-problem">⚡ {problem}</div>
                  <div class="lq-original">✗ &nbsp;<b>Found in your resume:</b><br>&nbsp;&nbsp;{original}</div>
                  <div class="lq-arrow">↓ how to fix it</div>
                  <div class="lq-rewritten">{rewrite_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()

# ── EXTRACTED TEXT PREVIEW ──────────────────────────────────────────────────

with st.expander("📄 Raw Extracted Resume Text (preview)"):
    st.text_area(
        "Extracted Text",
        resume_text[:3000] + ("…" if len(resume_text) > 3000 else ""),
        height=250,
        label_visibility="collapsed",
    )

# ── FOOTER ──────────────────────────────────────────────────────────────────

    st.caption(
        "CareerFit AI · Built with Python, Streamlit, scikit-learn & Plotly · "
        "For educational and placement preparation purposes."
    )

