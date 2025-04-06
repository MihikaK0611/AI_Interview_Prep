import cv2
import mediapipe as mp
import json
import time
import pyaudio
import numpy as np
import os
import random
import requests
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for


from vosk import Model, KaldiRecognizer
from deepface import DeepFace
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from gtts import gTTS
import pygame
from groq import Groq
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from datetime import timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import re
from collections import Counter
import string

app = Flask(__name__)

client = Groq(api_key="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")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load Vosk Speech Recognition Model
MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Vosk model not found at {MODEL_PATH}.")

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

# Initialize audio input
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=16384)
stream.start_stream()

# Initialize face detection & sentiment analysis
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
analyzer = SentimentIntensityAnalyzer()

# Initialize sentence transformer for similarity
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

qna_pairs = []




app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prompts.db'
app.config['SECRET_KEY'] = 'your_secret_key'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Prompt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prompt_text = db.Column(db.String(500), nullable=False)
    
    def __init__(self, prompt_text):
        self.prompt_text = prompt_text

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # 30-minute session timeout
login_manager.session_protection = "strong"

# Validate password
def validate_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r"[0-9]", password):
        return "Password must contain at least one digit."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return "Password must contain at least one special character."
    return None


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    global qna_pairs
    data = request.json
    role = data.get("role")
    interview_type = data.get("interview_type")
    experience = data.get("experience")
    skills = data.get("skills")

    prompt = f"""
    Generate 5 interview questions and their ideal answers for a {role} role in a {interview_type} interview.
    The candidate has {experience} years of experience and skills in {skills}.
    Provide the output strictly in JSON format like this:
    [
        {{"question": "Question 1?", "answer": "Ideal answer 1"}},
        {{"question": "Question 2?", "answer": "Ideal answer 2"}}
    ]
    """

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )

        raw_response = response.choices[0].message.content.strip()
        qna_pairs = json.loads(raw_response)
        return jsonify(qna_pairs)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/speak', methods=['POST'])
def speak():
    data = request.json
    text = data.get("text")
    filename = f"question_{random.randint(1000,9999)}.mp3"

    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    return jsonify({"audio": filename})

def semantic_similarity_score(user_answer, ideal_answer):
    embeddings = sentence_model.encode([user_answer, ideal_answer], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity * 100)

question_index = 0
responses = []
recording = False
emotion_history = []
stress_levels = []

eye_contact_count = 0
total_frames = 0

@app.route("/start_recording", methods=["POST"])
def start_recording():
    global question_index, recording, responses, eye_contact_count, total_frames, emotion_history, stress_levels, qna_pairs
    global emotional_stability_score, eye_contact_score, stress_level
    data = request.json
    role = data.get("role")
    interview_type = data.get("interview_type")
    experience = data.get("experience")
    skills = data.get("skills")

    # Generate questions
    response = requests.post("http://127.0.0.1:5000/generate_questions", json={
        "role": role,
        "interview_type": interview_type,
        "experience": experience,
        "skills": skills
    })
    if response.status_code != 200:
        return jsonify({"error": "Failed to generate questions."})

    qna_pairs = response.json()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Unable to access webcam"})

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        emotion = "Unknown"
        if results.detections:
            eye_contact_count += 1
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Draw the green box around detected face
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                # Extract face region
                face_crop = frame[y:y+height, x:x+width]
                if face_crop.size != 0:
                    try:
                        analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                        if analysis:
                            emotion = analysis[0]['dominant_emotion']
                            emotion_history.append(emotion)
                            if emotion in ["fear", "sad", "angry"]:
                                stress_levels.append(1)
                            else:
                                stress_levels.append(0)
                    except:
                        pass

        # Compute emotional stability score dynamically
        emotional_stability_score = round((emotion_history.count(max(set(emotion_history), key=emotion_history.count)) / len(emotion_history)) * 100) if emotion_history else 0

        # Display detected emotion & emotional stability score on webcam feed
        cv2.putText(frame, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Emotional Stability: {emotional_stability_score}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw a blue progress bar for emotional stability score
        bar_x = 20
        bar_y = 110
        bar_width = 200
        filled_width = int((emotional_stability_score / 100) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (200, 200, 200), -1)  # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + 20), (255, 0, 0), -1)  # Filled part

        if qna_pairs and 0 <= question_index < len(qna_pairs):
            question = qna_pairs[question_index].get("question", "No question found")
            cv2.putText(frame, f"Q: {question}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if recording:
            try:
                data = stream.read(4096, exception_on_overflow=False)
            except OSError:
                data = b""

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text_result = result["text"]

                if text_result.strip():
                    sentiment = analyzer.polarity_scores(text_result)
                    tone = "Neutral"
                    if sentiment["compound"] >= 0.05:
                        tone = "Positive"
                    elif sentiment["compound"] <= -0.05:
                        tone = "Negative"
                    confidence_score = round(abs(sentiment["compound"]) * 100)
                    confidence = "High" if confidence_score > 60 else "Medium" if confidence_score > 30 else "Low"

                    word_count = len(text_result.split())
                    clarity = "Detailed and Well-Explained" if word_count > 20 else "Needs more elaboration"

                    ideal_answer = qna_pairs[question_index]['answer']
                    similarity_score = semantic_similarity_score(text_result, ideal_answer)
                    similarity = "Highly Relevant" if similarity_score >= 75 else "Partially Relevant" if similarity_score >= 50 else "Irrelevant"

                    responses.append({
                        "question": qna_pairs[question_index]['question'],
                        "answer": text_result,
                        "correctness": similarity,
                        "confidence": confidence,
                        "tone": tone, 
                        "clarity": clarity,
                        "emotion": emotion
                    })

                    question_index += 1
                    recording = False  # Stop recording until 's' is pressed again

        cv2.imshow("AI Interview Coach", frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and not recording and question_index < len(qna_pairs):  
            # Speak the question before recording starts
            requests.post("http://127.0.0.1:5000/speak", json={"text": qna_pairs[question_index]['question']})
            recording = True  

        elif key == ord('q') or question_index >= len(qna_pairs):
            break

    cap.release()
    cv2.destroyAllWindows()
    stream.stop_stream()
    stream.close()
    audio.terminate()

    eye_contact_score = round((eye_contact_count / total_frames) * 100)
    stress_level = round((sum(stress_levels) / len(stress_levels)) * 100)

    return jsonify({
        "soft_skills": {
            "Eye Contact Score": eye_contact_score,
            "Emotional Stability Score": emotional_stability_score,
            "Stress Level": stress_level
        },
        "responses": responses
    })


@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    interview_data = {
        "soft_skills": {
            "Eye Contact Score": eye_contact_score,
            "Emotional Stability Score": emotional_stability_score,
            "Stress Level": stress_level
        },
        "responses": responses
    }
    return jsonify(interview_data)

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interview Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                padding: 20px;
                text-align: center;
            }}
            .container {{
                background: white;
                padding: 20px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                border-radius: 5px;
                max-width: 800px;
                margin: auto;
            }}
            h2 {{
                color: #333;
            }}
            .report-section {{
                text-align: left;
                margin-bottom: 20px;
            }}
            .question {{
                font-weight: bold;
                margin-top: 10px;
            }}
            .answer {{
                background: #e9ecef;
                padding: 10px;
                border-radius: 5px;
            }}
            .soft-skills {{
                background: #d4edda;
                padding: 10px;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Interview Report</h2>

            <div class="report-section">
                <h3>Soft Skills Analysis</h3>
                <div class="soft-skills">
                    <p><strong>Eye Contact Score:</strong> {data["soft_skills"]["Eye Contact Score"]}%</p>
                    <p><strong>Emotional Stability Score:</strong> {data["soft_skills"]["Emotional Stability Score"]}%</p>
                    <p><strong>Stress Level:</strong> {data["soft_skills"]["Stress Level"]}%</p>
                </div>
            </div>

            <div class="report-section">
                <h3>Question & Answer Review</h3>
                {"".join(f'<p class="question">{i+1}. {resp["question"]}</p><p class="answer"><strong>Answer:</strong> {resp["answer"]}</p><p><strong>Correctness:</strong> {resp["correctness"]}</p><p><strong>Confidence:</strong> {resp["confidence"]}</p><p><strong>Tone:</strong> {resp["tone"]}</p><p><strong>Clarity:</strong> {resp["clarity"]}</p><p><strong>Emotion Detected:</strong> {resp["emotion"]}</p><hr>' for i, resp in enumerate(data["responses"]))}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content



 
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Validate the password
        validation_error = validate_password(password)
        if validation_error:
            flash(validation_error, 'error')
            return redirect(url_for('signup'))

        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email is already in use.', 'error')
            return redirect(url_for('signup'))

        # Create a new user with hashed password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully!')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))   

if __name__ == "__main__":
    app.run(debug=True)
