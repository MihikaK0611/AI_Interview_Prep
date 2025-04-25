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
        return jsonify({"error": "Resume file not provided"}), 400

    file = request.files['resume']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Extract resume text
    resume_text = extract_text(filepath)

    # Get optional job description text
    job_description = request.form.get("jd", "")

    # Run analysis modules
    ats = calculate_ats_score(resume_text, job_description)
    grammar_issues = grammar_check(resume_text)
    readability = readability_score(resume_text)
    soft_skills = detect_soft_skills(resume_text)
    weak_phrases = detect_passive_phrases(resume_text)
    section_scores = rank_resume_sections(resume_text)

    return jsonify({
        "ATS Score": ats,
        "Grammar Issues": grammar_issues,
        "Readability Score": readability,
        "Soft Skills": soft_skills,
        "Weak Phrases": weak_phrases,
        "Section Scores": section_scores
    })

if __name__ == '__main__':
    app.run(debug=True)
