# utils/section_rank.py
import re

def section_score(text, section_keywords):
    return sum(1 for kw in section_keywords if kw.lower() in text.lower())

def rank_resume_sections(text):
    sections = {
        "Education": ["degree", "bachelor", "master", "university"],
        "Projects": ["project", "developed", "built", "created"],
        "Skills": ["python", "machine learning", "sql", "excel"]
    }
    scores = {section: section_score(text, kws) for section, kws in sections.items()}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
