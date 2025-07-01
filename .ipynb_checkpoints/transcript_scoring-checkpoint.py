import os
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(MODEL_NAME)

def score_transcripts(job_desc_path, transcript_dir):
    with open(job_desc_path, 'r', encoding='utf-8', errors='replace') as f:
        job_desc = f.read()
    job_embedding = embedder.encode(job_desc, convert_to_tensor=True)

    scores = {}
    for file in os.listdir(transcript_dir):
        if file.endswith(".txt"):
            path = os.path.join(transcript_dir, file)
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                transcript = f.read()
            transcript_embedding = embedder.encode(transcript, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(job_embedding, transcript_embedding).item()
            scores[file] = similarity
            print(f"{file} → similarity score: {similarity:.2f}")
    return scores
