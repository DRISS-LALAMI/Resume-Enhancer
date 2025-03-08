from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from transformers import pipeline
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Standalone functions
def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def extract_keywords(text):
    """Extract keywords using a Hugging Face model."""
    summarizer = pipeline("summarization", model="t5-small")
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]["summary_text"]



def generate_cover_letter(resume_text):
    """Generate a cover letter using a text generation model."""
    generator = pipeline("text-generation", model="gpt2")
    prompt = f"Write a professional cover letter based on the following resume:\n{resume_text}"
    cover_letter = generator(prompt, max_new_tokens=200, num_return_sequences=1)
    return cover_letter[0]["generated_text"]


def score_compatibility(resume_text, job_description):
    """Score compatibility between resume and job description."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0] * 100


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
        self.ner_pipeline = pipeline("ner", model="distilbert-base-uncased", grouped_entities=True)

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