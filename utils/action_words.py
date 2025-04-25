# utils/action_words.py
PASSIVE_PHRASES = ["was", "were", "is being", "are being", "have been", "has been"]

def detect_passive_phrases(text):
    return [phrase for phrase in PASSIVE_PHRASES if phrase in text.lower()]