"""
CareerFit AI - PDF Extraction Module
Extracts raw text from uploaded PDF resumes.
Falls back to PyPDF2 if pdfplumber is unavailable.
"""

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Accept a Streamlit UploadedFile object (or any file-like object).
    Returns extracted plain text string.
    Raises ValueError if extraction fails or returns empty text.
    """
    text = ""

    # Primary: pdfplumber (better text extraction)
    try:
        import pdfplumber
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except Exception:
        pass  # fall through to PyPDF2

    # Fallback: PyPDF2
    try:
        import PyPDF2
        # Reset file pointer for re-read
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            return text
    except Exception as e:
        raise ValueError(f"PDF extraction failed: {e}")

    raise ValueError(
        "Could not extract text from this PDF. "
        "Make sure the PDF is not scanned/image-based."
    )
