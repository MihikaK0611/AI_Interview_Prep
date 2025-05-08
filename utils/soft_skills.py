# utils/softskills.py
import ast

from groq import Groq
client = Groq(api_key="")


import re

def detect_soft_skills(summary_text):
    prompt = f"""
    Identify and list soft skills mentioned in this resume summary:
    \"{summary_text}\"
    Return only the list of soft skills — one per line or as a Python list.
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    raw = response.choices[0].message.content.strip()

    # Try different strategies to normalize the response
    cleaned = []

    # Try to match a Python-style list
    match_list = re.findall(r'"(.*?)"|\'(.*?)\'', raw)
    if match_list:
        cleaned = [item[0] or item[1] for item in match_list]

    # Fallback: split by lines or bullets
    if not cleaned:
        lines = raw.splitlines()
        for line in lines:
            line = line.strip("-•* \t\n")
            if ',' in line:
                cleaned.extend([item.strip() for item in line.split(',') if item.strip()])
            elif line:
                cleaned.append(line)

    # Final clean-up
    cleaned = [skill for skill in cleaned if len(skill) > 1 and not skill.lower().startswith("no soft")]

    return cleaned
