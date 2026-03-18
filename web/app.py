import streamlit as st
import requests
import pdfplumber
import fitz
import re

API_URL = "http://127.0.0.1:8000/predict/"

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_text_from_pdf(uploaded_file):
    try:
        text = ""
        uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if len(text.strip()) > 100:
            return clean_text(text)
    except Exception as e:
        st.warning(f"pdfplumber failed: {e}")

    try:
        uploaded_file.seek(0)
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([page.get_text() + "\n" for page in doc])
        return clean_text(text)
    except Exception as e:
        st.warning(f"PyMuPDF failed: {e}")

    return ""

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title("Resume Job Title Predictor")

uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
        if not st.session_state.resume_text:
            st.error("Failed to extract text from PDF.")
    elif uploaded_file.type == "text/plain":
        st.session_state.resume_text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type")

if st.session_state.resume_text:
    st.subheader("Extracted Resume Text:")
    st.text_area("Resume Content", value=st.session_state.resume_text, height=300, key="resume_display")

if st.button("Predict Job Titles"):
    if not st.session_state.resume_text.strip():
        st.error("No resume text to predict.")
    else:
        with st.spinner("Predicting..."):
            try:
                response = requests.post(API_URL, json={"resume_text": st.session_state.resume_text})
                if response.status_code == 200:
                    top5 = response.json().get("top5_job_titles", [])
                    if top5:
                        st.success("Top 5 Predicted Job Titles:")
                        for i, title in enumerate(top5, 1):
                            st.write(f"{i}. **{title}**")
                    else:
                        st.warning("No predictions returned.")
                else:
                    st.error(f"API error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
