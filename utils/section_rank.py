import re

def section_score(text, section_keywords):
    return sum(1 for kw in section_keywords if kw.lower() in text.lower())

def rank_resume_sections(text):
    sections = {
        "Education": ["degree", "bachelor", "master", "university", "gpa", "institute"],
        "Projects": ["project", "developed", "built", "created", "implemented"],
        "Skills": ["python", "machine learning", "sql", "excel", "tensorflow", "pandas"]
    }
    raw_scores = {section: section_score(text, kws) for section, kws in sections.items()}

    # Normalize to a score out of 10
    max_score = max(raw_scores.values()) or 1  # prevent div by zero
    scores = {section: round((score / max_score) * 10, 1) for section, score in raw_scores.items()}

    return scores
