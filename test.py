import cv2
import mediapipe as mp
import json
import time
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer
from deepface import DeepFace
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Vosk model not found at {MODEL_PATH}. Please download from https://alphacephei.com/vosk/models")
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=65536)
stream.start_stream()

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
analyzer = SentimentIntensityAnalyzer()

question_bank = {
    "Technology": [
        {"question": "What is Object-Oriented Programming?", "answer": "OOP is a programming paradigm using objects and classes."},
        {"question": "Explain the concept of Machine Learning.", "answer": "Machine learning is a branch of AI that allows computers to learn from data."},
        {"question": "How does a database work?", "answer": "A database stores structured information that can be retrieved and manipulated using queries."}
    ],
    "Marketing": [
        {"question": "What is digital marketing?", "answer": "Digital marketing is advertising through digital channels like social media, SEO, and email."},
        {"question": "Explain brand positioning.", "answer": "Brand positioning defines how a brand is perceived relative to competitors in the market."},
        {"question": "How do you measure marketing success?", "answer": "Marketing success is measured through KPIs like engagement, conversions, and revenue growth."}
    ],
    "Finance": [
        {"question": "What is the difference between assets and liabilities?", "answer": "Assets bring value to a company, while liabilities represent obligations."},
        {"question": "Explain the concept of compound interest.", "answer": "Compound interest is the interest calculated on both the initial principal and accumulated interest."},
        {"question": "What are the key financial statements?", "answer": "Key financial statements include the balance sheet, income statement, and cash flow statement."}
    ]
}

domain = input("Select a domain (Technology / Marketing / Finance): ").strip().title()
if domain not in question_bank:
    print("Invalid domain. Defaulting to Technology.")
    domain = "Technology"

questions = question_bank[domain]
random.shuffle(questions)

cap = cv2.VideoCapture(0)
print("AI Interview Coach Running... Press 's' to start speaking and 'q' to exit.")
feedback = ""
last_feedback_time = time.time()
question_index = 0
responses = []
recording = False
emotion_history = []
stress_levels = []

eye_contact_count = 0
total_frames = 0
engagement_score = 0

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity_score(user_answer, ideal_answer):
    embeddings = model.encode([user_answer, ideal_answer], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity * 100)  # Convert similarity to percentage

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
            height = int(bbox.height * h) if hasattr(bbox, 'height') else width

            face_crop = frame[y:y+height, x:x+width]
            if face_crop.size != 0:
                try:
                    analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion']
                    emotion_history.append(emotion)
                    
                    if emotion in ["fear", "sad", "angry"]:
                        stress_levels.append(1)
                    else:
                        stress_levels.append(0)

                except:
                    pass

            cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    if question_index < len(questions):
        cv2.putText(frame, f"Q: {questions[question_index]['question']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
                clarity = "Detailed and Well-Explained" if word_count > 20 else "Satisfactory but could be improved" if word_count >= 10 else "Needs more elaboration"

                ideal_answer = questions[question_index]['answer']
                similarity_score = semantic_similarity_score(text_result, ideal_answer)  # Using Sentence Transformers
                if similarity_score >= 75:
                    similarity = "Highly Relevant "
                elif similarity_score >= 50:
                    similarity = "Partially Relevant"
                else:
                    similarity = "Irrelevant"
                feedback = f"Tone: {tone}, Confidence: {confidence}, Answer Quality: {clarity}, Relevance: {similarity}"
                last_feedback_time = time.time()
                responses.append({
                    "question": questions[question_index]['question'],
                    "answer": text_result,
                    "correctness": similarity,
                    "confidence": confidence,
                    "tone": tone, 
                    "clarity": clarity,
                    "emotion": emotion
                })

                question_index += 1
                recording = False

    if time.time() - last_feedback_time < 5:
        cv2.putText(frame, f"Feedback: {feedback}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("AI Interview Coach", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        recording = True

    elif key == ord('q') or question_index >= len(questions):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
audio.terminate()

eye_contact_score = round((eye_contact_count / total_frames) * 100)
emotional_stability_score = round((emotion_history.count(max(set(emotion_history), key=emotion_history.count)) / len(emotion_history)) * 100)
stress_level = round((sum(stress_levels) / len(stress_levels)) * 100)

report = {
    "soft_skills": {
        "Eye Contact Score": eye_contact_score,
        "Emotional Stability Score": emotional_stability_score,
        "Stress Level": stress_level,
        "Speaking Clarity": clarity
    },
    "responses": responses
}

with open("interview_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Interview Completed! Report saved as interview_report.json")
