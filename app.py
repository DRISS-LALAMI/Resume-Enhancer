import streamlit as st
from resume_enhancer import ResumeEnhancer, JobMatcher
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Initialize classes
resume_enhancer = ResumeEnhancer(OPENAI_API_KEY)
job_matcher = JobMatcher(OPENAI_API_KEY)

# Streamlit App
st.title("AI-Powered Resume Enhancer")

# Input fields
resume = st.text_area("Paste your resume here:")
job_description = st.text_area("Paste the job description here (optional):")

if st.button("Enhance Resume"):
    if resume:
        # Extract entities
        entities = resume_enhancer.extract_entities(resume)
        st.write("Extracted Entities:")
        st.write(entities)

        # Improve resume
        improved_resume = resume_enhancer.improve_resume(resume)
        st.write("Improved Resume:")
        st.write(improved_resume)

        # Generate cover letter
        cover_letter = resume_enhancer.generate_cover_letter(resume)
        st.write("Generated Cover Letter:")
        st.write(cover_letter)

        # Match resume to job description (if provided)
        if job_description:
            match_score = job_matcher.match_resume_to_job(resume, job_description)
            st.write(f"Job Match Score: {match_score}")
    else:
        st.error("Please paste your resume to proceed.")