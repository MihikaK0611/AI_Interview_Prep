# utils/softskills.py
from groq import Groq
client = Groq(api_key="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")

def detect_soft_skills(summary_text):
    prompt = f"""
    Identify and list soft skills mentioned in this resume summary:
    "{summary_text}"
    Return them as a list.
    """
    response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
    return response.choices[0].message.content.strip()