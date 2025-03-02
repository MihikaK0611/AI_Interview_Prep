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
from groq import Groq

# Initialize Groq client
client = Groq(api_key="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")
  # Ensure you have set this env variable

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load Vosk speech recognition model
MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Vosk model not found at {MODEL_PATH}. Download from https://alphacephei.com/vosk/models")

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

# Initialize audio input
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
stream.start_stream()

# Initialize face detection & sentiment analysis
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
analyzer = SentimentIntensityAnalyzer()
def generate_questions_and_answers(role, interview_type, experience, skills):
    prompt = f"""
Generate 5 interview questions and their ideal answers for a {role} role in a {interview_type} interview.
The candidate has {experience} years of experience and skills in {skills}.
Provide the output strictly in JSON format like this:
[
    {{"question": "Question 1?", "answer": "Ideal answer 1"}},
    {{"question": "Question 2?", "answer": "Ideal answer 2"}}
]
"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="mixtral-8x7b-32768",
    )

    # ✅ Debugging: Print raw API response
    if not response.choices or not response.choices[0].message.content.strip():
        print("⚠️ Error: API returned an empty response.")
        return []

    raw_response = response.choices[0].message.content.strip()
    print("🔍 Raw API Response:", raw_response)  # Debugging step

    try:
        # ✅ Clean and check JSON format
        cleaned_response = raw_response.replace("\n", "").replace("\t", "").strip()

        if not cleaned_response.startswith("[") or not cleaned_response.endswith("]"):
            print("⚠️ Error: API response is not in valid JSON format.")
            return []

        # ✅ Parse JSON safely
        qna_pairs = json.loads(cleaned_response)
        return qna_pairs

    except json.JSONDecodeError as e:
        print("❌ Parsing Error:", str(e))
        return []

# Get user input for interview settings
role = input("Enter the role (e.g., Software Engineer, Marketing Manager): ").strip()
interview_type = input("Enter the interview type (Technical/HR/Behavioral): ").strip()
skills = input("Enter the skills (e.g., Web Development, DSA,AI/ML): ").strip()
Workexperience = input("years of experience in particular Role you have Selected ").strip()


qna_pairs = generate_questions_and_answers(role, interview_type,skills,Workexperience)
if not qna_pairs:
    print("Failed to fetch questions. Exiting...")
    exit()

random.shuffle(qna_pairs)  # Randomize question order

# Open webcam for AI interview coaching
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

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to calculate semantic similarity
def semantic_similarity_score(user_answer, ideal_answer):
    print()
    embeddings = sentence_model.encode([user_answer, ideal_answer], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity * 100)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    emotion = "Unknown"
    if results.detections:
        eye_contact_count += 1  # User is making eye contact
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
                    if analysis:
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

    if question_index < len(qna_pairs):
        cv2.putText(frame, f"Q: {qna_pairs[question_index]['question']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

                feedback = f"Tone: {tone}, Confidence: {confidence}, Answer Quality: {clarity}, Relevance: {similarity}"
                last_feedback_time = time.time()

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
                recording = False

    if time.time() - last_feedback_time < 5:
        cv2.putText(frame, f"Feedback: {feedback}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("AI Interview Coach", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        recording = True

    elif key == ord('q') or question_index >= len(qna_pairs):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
audio.terminate()


eye_contact_score = round((eye_contact_count / total_frames) * 100)
emotional_stability_score = round((emotion_history.count(max(set(emotion_history), key=emotion_history.count)) / len(emotion_history)) * 100)
stress_level = round((sum(stress_levels) / len(stress_levels)) * 100)

# Generate final report
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
