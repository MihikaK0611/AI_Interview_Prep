# utils/grammar.py
import language_tool_python
import textstat

tool = language_tool_python.LanguageTool('en-US')

from collections import Counter

def grammar_check(text, top_n=10):
    matches = tool.check(text)
    all_messages = [match.message.strip() for match in matches]
    message_counts = Counter(all_messages)
    # Return sorted top N most common issues
    return [f"{msg} ({count} times)" for msg, count in message_counts.most_common(top_n)]


def readability_score(text):
    return textstat.flesch_reading_ease(text)
