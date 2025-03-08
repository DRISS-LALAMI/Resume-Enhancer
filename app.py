import streamlit as st
from resume_enhancer import ResumeEnhancer, JobMatcher, extract_text_from_pdf, extract_keywords, generate_cover_letter, score_compatibility
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize classes
resume_enhancer = ResumeEnhancer(OPENAI_API_KEY)
job_matcher = JobMatcher(OPENAI_API_KEY)

# Streamlit App
st.title("AI-Powered Resume Enhancer")

# Input options
input_option = st.radio("How would you like to input your resume?", ("Upload PDF", "Paste Text"))

resume_text = ""
if input_option == "Upload PDF":
    # Upload resume as PDF
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.write("Resume text extracted successfully!")
else:
    # Paste resume as text
    resume_text = st.text_area("Paste your resume here:")

if resume_text:
    # Options for enhancing the resume
    option = st.selectbox(
        "What would you like to do?",
        ("Extract Keywords", "Generate Cover Letter", "Score Compatibility with Job Description")
    )

    if option == "Extract Keywords":
        keywords = extract_keywords(resume_text)
        st.write("Extracted Keywords:")
        st.write(keywords)

    elif option == "Generate Cover Letter":
        cover_letter = generate_cover_letter(resume_text)
        st.write("Generated Cover Letter:")
        st.write(cover_letter)

    elif option == "Score Compatibility with Job Description":
        job_description = st.text_area("Paste the job description here:")
        if job_description:
            score = score_compatibility(resume_text, job_description)
            st.write(f"Compatibility Score: {score:.2f}%")