import os
from sentence_transformers import SentenceTransformer, util
from docx import Document
import fitz
from io import BytesIO

MODEL_NAME = 'all-mpnet-base-v2'
embedder = SentenceTransformer(MODEL_NAME, "cache_folder=/root/.cache")

def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.endswith('.docx'):
        doc = Document(BytesIO(file_bytes))
        return '\n'.join(p.text for p in doc.paragraphs)

    elif filename.endswith('.pdf'):
        text = ''
        pdf = fitz.open(stream=file_bytes, filetype='pdf')
        for page in pdf:
            text += page.get_text()
        return text

    elif filename.endswith('.txt'):
        return file_bytes.decode('utf-8', errors='replace')

    else:
        raise ValueError("Unsupported file type")


def score_resume_file(resume_text, job_text):
    """
    Compute cosine similarity score between a single resume and job description.
    """
    job_embedding = embedder.encode(job_text, convert_to_tensor=True)
    resume_embedding = embedder.encode(resume_text, convert_to_tensor=True)
    return util.pytorch_cos_sim(job_embedding, resume_embedding).item()

# Add the missing function that main.py is calling
def compute_cosine_score(resume_text, job_text):
    """
    Compute cosine similarity score between resume text and job description text.
    This function works directly with text content (not file paths).
    """
    return score_resume_file(resume_text, job_text)