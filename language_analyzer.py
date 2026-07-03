"""
CareerFit AI - Language Quality Analyzer
Powered by Groq API (FREE) using the official groq Python package.

HOW TO GET YOUR FREE GROQ API KEY:
  1. Go to https://console.groq.com
  2. Sign up with Google or email (no credit card needed)
  3. Click "API Keys" -> "Create API Key"
  4. Open .streamlit/secrets.toml and replace:
     GROQ_API_KEY = "gsk_your_key_here"
     with your actual key
  5. Save and restart the app
"""

import re
import json
import os


def _get_groq_key() -> str:
    """Get Groq API key from Streamlit secrets or environment variable."""
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key and "your_key" not in key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def analyze_language_quality(resume_text: str) -> dict:
    """
    Analyze resume text using Groq API (Llama 3).
    Returns dict with: summary, human_score, issues, priority_counts, mode.
    """
    import groq as groq_sdk

    api_key = _get_groq_key()

    if not api_key:
        raise ValueError(
            "Groq API key not found.\n\n"
            "Steps to fix:\n"
            "1. Go to https://console.groq.com and sign up free\n"
            "2. Click API Keys → Create API Key\n"
            "3. Open the file: .streamlit/secrets.toml\n"
            "4. Replace: GROQ_API_KEY = \"gsk_your_key_here\"\n"
            "   with your actual key\n"
            "5. Save the file and restart the app"
        )

    system_prompt = """You are a professional resume language expert with 10+ years of experience 
reviewing resumes for top tech companies. You know exactly what recruiters look for 
and what gets resumes rejected.

Your job: analyze the resume and find ONLY genuine language problems.

STRICT RULES — follow every single one:
1. Soft skills listed under a "Soft Skills" or "Skills" section (e.g. Team Player, 
   Communication, Problem Solving, Adaptability) are COMPLETELY ACCEPTABLE — do NOT flag them.
2. Only flag "Team Player", "Communication" etc. if they appear as unsupported claims 
   in a SUMMARY or OBJECTIVE line with zero evidence.
3. Every rewritten suggestion MUST be specific to the exact sentence the user wrote — 
   not a generic template. Use the actual role, tools, numbers from their resume.
4. Never repeat the same advice twice across different issues.
5. Use the CAR formula: Context -> Action -> Result with numbers wherever possible.
6. Do NOT flag things that are standard, acceptable resume writing.

THREE PROBLEM TYPES:
- ai_sounding: Generic buzzwords, template phrases, unverifiable claims in summary/objective
- weak_language: Passive verbs, vague skill claims, achievements without measurable numbers
- unprofessional: First-person bullets (I built, I did), informal words (stuff, kind of, a bit)

THREE PRIORITY LEVELS:
- high: Directly reduces shortlisting chances (passive verbs, no numbers, first-person bullets)
- medium: Weakens profile (vague skills, mild buzzwords)
- low: Minor polish (redundant words, small informal phrases)

RESPONSE FORMAT — return ONLY valid JSON, nothing else, no markdown fences:
{
  "summary": "2 sentences: biggest strength AND biggest weakness you found",
  "human_score": <integer 0-100>,
  "issues": [
    {
      "original": "exact sentence or phrase copied from the resume as-is",
      "problem": "specific explanation mentioning what is missing from THIS sentence",
      "type": "weak_language",
      "priority": "high",
      "rewritten": "improved version using the actual tools/role/context from their resume"
    }
  ]
}"""

    user_prompt = f"""Analyze this resume carefully.
Remember: soft skills in a Skills section are ACCEPTABLE — do not flag them.
Make every rewritten suggestion specific to what this person actually wrote.

RESUME:
{resume_text[:4000]}

Return only valid JSON. No markdown, no explanation outside the JSON."""

    try:
        client = groq_sdk.Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.3,
        )
    except groq_sdk.AuthenticationError:
        raise ValueError(
            "Invalid Groq API key.\n"
            "Open .streamlit/secrets.toml and make sure your key is correct.\n"
            "Get a valid key at https://console.groq.com"
        )
    except groq_sdk.RateLimitError:
        raise ValueError(
            "Groq rate limit reached. Wait a few seconds and try again.\n"
            "Free tier allows 30 requests/minute."
        )
    except Exception as e:
        raise ValueError(f"Groq API error: {str(e)}")

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model accidentally added them
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$",          "", raw).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON block from response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError(
                "Groq returned an unexpected response. Please click the button again."
            )

    # Normalize all fields
    result.setdefault("summary",     "Analysis complete.")
    result.setdefault("human_score", 50)
    result.setdefault("issues",      [])
    result["human_score"] = max(0, min(100, int(result["human_score"])))

    valid_types      = {"ai_sounding", "weak_language", "unprofessional"}
    valid_priorities = {"high", "medium", "low"}

    for issue in result["issues"]:
        issue.setdefault("original",  "")
        issue.setdefault("problem",   "")
        issue.setdefault("rewritten", "")
        issue.setdefault("priority",  "medium")
        if issue["type"]     not in valid_types:      issue["type"]     = "weak_language"
        if issue["priority"] not in valid_priorities: issue["priority"] = "medium"

    # Build priority counts
    pc = {"high": 0, "medium": 0, "low": 0}
    for issue in result["issues"]:
        pc[issue["priority"]] += 1

    result["priority_counts"] = pc
    result["mode"] = "ai"

    return result
