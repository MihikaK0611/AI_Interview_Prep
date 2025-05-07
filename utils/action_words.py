# utils/action_words.py
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def detect_passive_phrases(text):
    matches = tool.check(text)
    passive_phrases = []

    for match in matches:
        if "passive voice" in match.ruleId.lower() or "passive voice" in match.message.lower():
            passive_phrases.append(match.context)

    return list(set(passive_phrases))
