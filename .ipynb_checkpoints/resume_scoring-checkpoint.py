import os
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(MODEL_NAME)

def score_resumes(job_desc_path, resume_dir):
    with open(job_desc_path, 'r', encoding='utf-8', errors='replace') as f:
        job_desc = f.read()
    job_embedding = embedder.encode(job_desc, convert_to_tensor=True)

    scores = {}
    for file in os.listdir(resume_dir):
        if file.endswith(".txt"):
            path = os.path.join(resume_dir, file)
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                resume = f.read()
            resume_embedding = embedder.encode(resume, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(job_embedding, resume_embedding).item()
            scores[file] = similarity
            print(f"{file} → similarity score: {similarity:.2f}")
    return scores
