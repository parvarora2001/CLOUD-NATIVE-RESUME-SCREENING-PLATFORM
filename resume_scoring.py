import os
from sentence_transformers import SentenceTransformer, util
from docx import Document
import fitz  

MODEL_NAME = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(MODEL_NAME)

def read_docx(path):
    doc = Document(path)
    return '\n'.join(para.text for para in doc.paragraphs)

def read_pdf(path):
    doc = fitz.open(path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text

def read_resume(path):
    if path.endswith(".pdf"):
        return read_pdf(path)
    elif path.endswith(".docx"):
        return read_docx(path)
    elif path.endswith(".txt"):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def score_resume(job_desc_path, resume_dir):
    with open(job_desc_path, 'r', encoding='utf-8', errors='replace') as f:
        job_desc = f.read()

    job_embedding = embedder.encode(job_desc, convert_to_tensor=True)

    scores = {}
    for file in os.listdir(resume_dir):
        full_path = os.path.join(resume_dir, file)
        if file.endswith(".txt"):
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                resume = f.read()
        elif file.endswith(".docx"):
            resume = read_docx(full_path)
        elif file.endswith(".pdf"):
            resume = read_pdf(full_path)
        else:
            continue  # Skip unsupported files

        resume_embedding = embedder.encode(resume, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(job_embedding, resume_embedding).item()
        scores[file] = similarity
        print(f"{file} → similarity score: {similarity:.2f}")

    return scores
    
