# app.py
import streamlit as st
import google.generativeai as genai
import PyPDF2 as pdf
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Page config ---
st.set_page_config(page_title="Gemini ATS Resume Builder", page_icon=":guardsman:", layout="wide")

# --- Custom Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Base font */
    html, body, [class*="css"]  {
        font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        color: #e6eef8;
    }

    /* App background */
    .stApp {
        background: linear-gradient(180deg, #0b1221 0%, #071024 100%);
        padding-top: 24px;
        padding-bottom: 40px;
    }

    /* Card style */
    .card {
        background: rgba(255,255,255,0.02);
        padding: 24px;
        border-radius: 14px;
        box-shadow: 0 10px 30px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* Title */
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    h2, h3, h4, h5, h6 {
        color: #cbd5e1 !important;
    }

    /* Text area */
    textarea[aria-label="Paste the Job Description here"] {
        background: rgba(255,255,255,0.05) !important;
        color: #e6eef8 !important;
        border-radius: 12px !important;
        padding: 14px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        min-height: 140px !important;
        font-size: 14px !important;
    }
    textarea[aria-label="Paste the Job Description here"]::placeholder {
        color: #94a3b8 !important;
        opacity: 1;
    }

    /* File uploader */
    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #cbd5e1 !important;
        padding: 12px !important;
    }

    /* Primary button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg,#0ea5a0,#06b6d4) !important;
        color: white !important;
        border: none !important;
        padding: 10px 18px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 26px rgba(6,22,37,0.6) !important;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 36px rgba(6,22,37,0.7) !important;
    }

    /* Result boxes */
    .result-box {
        background: rgba(255,255,255,0.04);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Labels */
    label, .stMarkdown, .stText, .stWrite {
        color: #cbd5e1 !important;
        font-weight: 500;
    }

    /* Spacing utility */
    .block {
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helper functions ---
def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

def clean_json_response(response):
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        json_str = json_str.replace('\n', ' ').replace('\r', '')
        return json_str
    return None

def input_pdf_text(file):
    reader = pdf.PdfReader(file)
    text = ""
    for p in range(len(reader.pages)):
        page = reader.pages[p]
        txt = page.extract_text()
        if txt:
            text += txt + "\n"
    return text

# --- Layout ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("Gemini ATS Resume Builder")
    st.markdown("#### Improve your Resume for ATS (Applicant Tracking Systems)")
    st.markdown("<div class='block'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        jd = st.text_area("Paste the Job Description here", placeholder="Paste full job description (responsibilities, skills, qualifications)...")
    with col2:
        uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf", help="Upload your resume in PDF format")

    st.markdown("<div class='block'></div>", unsafe_allow_html=True)

    submit = st.button("Analyze Resume")

    st.markdown('</div>', unsafe_allow_html=True)

# --- Logic ---
input_prompt = """ 
You are an expert ATS (Applicant Tracking System) with deep knowledge in tech fields including software engineering, data science, data analytics and machine learning.

Your task is to analyze the resume against the given job description and 
provide a deatiled analysis of how well the resume matches the job description.
consider the competitive job market and provide a actionable improvement suggestions .

IMPORTANT: You must respond with ONLY a JSON object in the following exact format, with no additional text:
{{
    "JD Match": "X%", 
    "MissingKeywords": ["keyword1", "keyword2", ...],
    "profileSummary": "your detailed analysis and improvement suggestions here"
}}

resume: {text}
description: {jd}
"""

if submit:
    if uploaded_file is not None and jd:
        with st.spinner("Analyzing resume — this may take a few seconds..."):
            text = input_pdf_text(uploaded_file)
            prompt = input_prompt.format(text=text, jd=jd)
            response = get_gemini_response(prompt)
            json_str = clean_json_response(response)

        if json_str:
            try:
                result = json.loads(json_str)
                st.markdown("<div class='card' style='margin-top:18px'>", unsafe_allow_html=True)
                st.subheader("Analysis Results")

                st.markdown(f"<div class='result-box'><strong>JD Match:</strong> {result.get('JD Match')}</div>", unsafe_allow_html=True)
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                st.markdown("<strong>Missing Keywords:</strong>", unsafe_allow_html=True)
                if isinstance(result.get('MissingKeywords'), list) and len(result['MissingKeywords']) > 0:
                    missing_md = "<ul>"
                    for kw in result['MissingKeywords']:
                        missing_md += f"<li style='color:#cbd5e1'>{kw}</li>"
                    missing_md += "</ul>"
                    st.markdown(f"<div class='result-box'>{missing_md}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result-box'>No missing keywords detected.</div>", unsafe_allow_html=True)

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                st.markdown("<strong>Profile Summary / Suggestions:</strong>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-box'>{result.get('profileSummary')}</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
            except json.JSONDecodeError:
                st.error("Error parsing the JSON response. Please check the model output.")
        else:
            st.error("No valid JSON response found. Please check the model output.")
    else:
        st.error("Please upload a PDF resume and provide a job description.")
