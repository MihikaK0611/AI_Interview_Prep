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

MODEL_PATH = "vosk-model-small-en-us-0.15"
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
        "What is Object-Oriented Programming?",
        "Explain the concept of Machine Learning.",
        "How does a database work?"
    ],
    "Marketing": [
        "What is digital marketing?",
        "Explain brand positioning.",
        "How do you measure marketing success?"
    ],
    "Finance": [
        "What is the difference between assets and liabilities?",
        "Explain the concept of compound interest.",
        "What are the key financial statements?"
    ]
}

domain = input("Select a domain (Technology / Marketing / Finance): ").strip().title()
if domain not in question_bank:
    print("Invalid domain. Defaulting to Technology.")
    domain = "Technology"

questions = question_bank[domain]
random.shuffle(questions)

cap = cv2.VideoCapture(0)
print("🎥 AI Interview Coach Running... Press 'q' to exit.")

feedback = ""
last_feedback_time = time.time()
question_index = 0
responses = []
recording = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    emotion = "Unknown"
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            face_crop = frame[y:y+height, x:x+width]
            if face_crop.size != 0:
                try:
                    analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion']
                except:
                    pass
            cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    if question_index < len(questions):
        cv2.putText(frame, f"Q: {questions[question_index]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
                quality = "Detailed and Well-Explained" if word_count > 20 else "Satisfactory but could be improved" if word_count >= 10 else "Needs more elaboration"

                feedback = f"Tone: {tone}, Confidence: {confidence}, Answer Quality: {quality}"
                last_feedback_time = time.time()

                responses.append({
                    "question": questions[question_index],
                    "answer": text_result,
                    "tone": tone,
                    "confidence": confidence,
                    "quality": quality,
                    "emotion": emotion
                })

                recording = False  # Stop recording **before moving to the next question**
                question_index += 1  # Only update after response is stored

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

report = {
    "domain": domain,
    "responses": responses,
}

with open("interview_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Interview completed! Report saved as interview_report.json")
