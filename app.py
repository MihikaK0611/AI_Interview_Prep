from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from utils.resume_parser import extract_text
from utils.ats_score import calculate_ats_score
from utils.grammar import grammar_check, readability_score
from utils.soft_skills import detect_soft_skills
from utils.action_words import detect_passive_phrases
from utils.section_rank import rank_resume_sections

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return "Resume Analyzer API is running. POST to /analyze_resume"

@app.route("/upload")
def upload_ui():
    return render_template("index2.html")

@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    if 'resume' not in request.files:
        return "Error: Resume file not provided.", 400, {'Content-Type': 'text/plain'}

    file = request.files['resume']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    resume_text = extract_text(filepath)
    job_description = request.form.get("jd", "")

    ats = calculate_ats_score(resume_text, job_description)
    grammar_issues = grammar_check(resume_text)
    readability = readability_score(resume_text)
    soft_skills = [s.strip() for s in detect_soft_skills(resume_text) if s.strip()]
    weak_phrases = detect_passive_phrases(resume_text)
    section_scores = rank_resume_sections(resume_text)

    # Normalize section_scores to dict if not already
    if isinstance(section_scores, list):
        section_scores = {f"Section {i+1}": val for i, val in enumerate(section_scores)}
    elif not isinstance(section_scores, dict):
        section_scores = {}

    return render_template("report.html",
                           ats=ats,
                           grammar_issues=grammar_issues if isinstance(grammar_issues, list) else [],
                           readability=readability,
                           soft_skills=soft_skills,
                           weak_phrases=weak_phrases,
                           section_scores=section_scores)


if __name__ == '__main__':
    app.run(debug=True)
