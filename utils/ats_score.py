# utils/ats_score.py
import spacy
nlp = spacy.load("en_core_web_sm")

def calculate_ats_score(resume_text, job_keywords):
    doc = nlp(resume_text.lower())
    matched_keywords = [kw for kw in job_keywords if kw.lower() in resume_text.lower()]
    return len(matched_keywords) / len(job_keywords) * 100 if job_keywords else 0