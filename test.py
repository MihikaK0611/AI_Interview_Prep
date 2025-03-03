import cv2 
import mediapipe as mp
import json
import time
import pyaudio
import numpy as np
import os
import random
from vosk import Model, KaldiRecognizer
from deepface import DeepFace
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from gtts import gTTS
import pygame
from groq import Groq


client = Groq(api_key="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")  


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load Vosk Speech Recognition Model
MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Vosk model not found at {MODEL_PATH}. Download from https://alphacephei.com/vosk/models")

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

# Function to generate AI-powered interview questions
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

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
        )

        raw_response = response.choices[0].message.content.strip()

        qna_pairs = json.loads(raw_response)
        return qna_pairs
    except json.JSONDecodeError:
        print("Error parsing API response! The response was not valid JSON.")
        return []
    except Exception as e:
        print(f"API call failed: {e}")
        return []


# Function to convert text to speech and play the audio
def speak(text):
    if os.path.exists("question.mp3"):
        try:
            pygame.mixer.quit()  # Ensure pygame releases the file
            os.remove("question.mp3")  # Delete the file before creating a new one
        except PermissionError:
            print("Warning: Unable to delete question.mp3, trying a new filename.")
            filename = f"question_{random.randint(1000,9999)}.mp3"  # Use a unique filename
        else:
            filename = "question.mp3"
    else:
        filename = "question.mp3"

    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)


# Get user input for interview settings
role = input("Enter the role (e.g., Software Engineer, Marketing Manager): ").strip()
interview_type = input("Enter the interview type (Technical/HR/Behavioral): ").strip()
skills = input("Enter the skills (e.g., Web Development, DSA, AI/ML): ").strip()
experience = input("Years of experience in the selected role: ").strip()

qna_pairs = generate_questions_and_answers(role, interview_type, experience, skills)
if not qna_pairs:
    print("Failed to fetch questions. Exiting...")
    exit()

random.shuffle(qna_pairs)

# Open webcam for AI interview coaching
cap = cv2.VideoCapture(0)
print("AI Interview Coach Running... Press 's' to start speaking and 'q' to exit.")

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
        eye_contact_count += 1
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

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
        speak(qna_pairs[question_index]['question'])  # Speak the question only when 's' is pressed
        recording = True  # Start recording the answer

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
