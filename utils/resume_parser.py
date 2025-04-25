# utils/resume_parser.py
import docx2txt
from PyPDF2 import PdfReader

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file_path.endswith('.docx'):
        return docx2txt.process(file_path)
    else:
        raise ValueError("Unsupported file format")


