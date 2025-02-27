import cv2
import mediapipe as mp
import json
import time
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer
from deepface import DeepFace
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = r"C:\Users\skbob\Downloads\vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Vosk model not found at {MODEL_PATH}. Please download from https://alphacephei.com/vosk/models")
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=32768)
stream.start_stream()

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
analyzer = SentimentIntensityAnalyzer()

question_bank = {
    "Technology": [
        {"question": "What is Object-Oriented Programming?", "ideal_answer": "Object-Oriented Programming is a paradigm that organizes data into objects with attributes and methods."},
        {"question": "Explain the concept of Machine Learning.", "ideal_answer": "Machine learning is a subset of AI that allows systems to learn from data without explicit programming."},
    ],
    "Marketing": [
        {"question": "What is digital marketing?", "ideal_answer": "Digital marketing refers to marketing efforts that use the internet, social media, search engines, and other online channels."},
        {"question": "Explain brand positioning.", "ideal_answer": "Brand positioning is the strategy of differentiating a brand in the market by targeting specific audiences."},
    ],
}

domain = input("Select a domain (Technology / Marketing): ").strip().title()
if domain not in question_bank:
    print("Invalid domain. Defaulting to Technology.")
    domain = "Technology"

questions = question_bank[domain]
random.shuffle(questions)

cap = cv2.VideoCapture(0)
print("AI Interview Coach Running... Press 's' to start speaking and 'q' to exit.")

emotion_history = []
stress_levels = []
responses = []
recording = False
question_index = 0

eye_contact_count = 0
total_frames = 0
engagement_score = 0

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
                confidence_score = round(abs(sentiment["compound"]) * 100)
                confidence = "High" if confidence_score > 60 else "Medium" if confidence_score > 30 else "Low"
                word_count = len(text_result.split())
                clarity = "Clear & Detailed" if word_count > 20 else "Needs More Explanation"

                ideal_answer = questions[question_index]['ideal_answer']
                correctness_score = round((len(set(text_result.split()) & set(ideal_answer.split())) / len(set(ideal_answer.split()))) * 100)

                responses.append({
                    "question": questions[question_index]['question'],
                    "answer": text_result,
                    "correctness_score": correctness_score,
                    "confidence": confidence,
                    "clarity": clarity,
                    "emotion": emotion
                })

                question_index += 1
                recording = False 

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
