# utils/grammar.py
import language_tool_python
import textstat

def grammar_check(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    grammar_errors = len(matches)
    return grammar_errors

def readability_score(text):
    readability_score = textstat.flesch_reading_ease(text)
    return readability_score