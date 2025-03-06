from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from transformers import pipeline

class ResumeEnhancer:
    def __init__(self, openai_api_key):
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
        except ImportError:
            raise ImportError("PyTorch is not installed. Please install it using `pip install torch`.")

        try:
            import tensorflow as tf
            print(f"TensorFlow version: {tf.__version__}")
        except ImportError:
            raise ImportError("TensorFlow is not installed. Please install it using `pip install tensorflow`.")

        self.llm = OpenAI(api_key=openai_api_key)
        self.ner_pipeline = pipeline("ner", model="bert-base-uncased")

    def extract_entities(self, resume_text):
        """Extract key entities (e.g., skills, job titles) from the resume."""
        entities = self.ner_pipeline(resume_text)
        return entities

    def improve_resume(self, resume_text):
        """Suggest improvements for the resume."""
        template = """
        Analyze the following resume and suggest improvements:
        {resume}
        """
        prompt = PromptTemplate(template=template, input_variables=["resume"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        improved_resume = chain.run(resume_text)
        return improved_resume

    def generate_cover_letter(self, resume_text):
        """Generate a cover letter based on the resume."""
        template = """
        Write a professional cover letter based on the following resume:
        {resume}
        """
        prompt = PromptTemplate(template=template, input_variables=["resume"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        cover_letter = chain.run(resume_text)
        return cover_letter


class JobMatcher:
    def __init__(self, openai_api_key):
        self.llm = OpenAI(api_key=openai_api_key)

    def match_resume_to_job(self, resume_text, job_description):
        """Compare the resume to a job description and provide a match score."""
        template = """
        Compare the following resume and job description, and provide a match score (0-100):
        Resume: {resume}
        Job Description: {job_description}
        """
        prompt = PromptTemplate(template=template, input_variables=["resume", "job_description"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        match_score = chain.run({"resume": resume_text, "job_description": job_description})
        return match_score