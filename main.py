# main.py
import os
import pandas as pd
from resume_scoring import score_resume, read_resume # type: ignore
from llm_explainer import explain_fit

JOB_DESC_PATH = 'data/job_description.txt'
RESUME_DIR = 'data/resumes/'

if __name__ == "__main__":
    print("Reading job description...")
    with open(JOB_DESC_PATH, "r", encoding="utf-8") as f:
        job_desc_text = f.read()

    print("\nScoring resumes...")
    resume_scores = score_resume(JOB_DESC_PATH, RESUME_DIR)

    print("\nGenerating explanations with Ollama...")
    results = []
    for filename, score in resume_scores.items():
        resume_path = os.path.join(RESUME_DIR, filename)
        resume_text = read_resume(resume_path)
        explanation = explain_fit(resume_text, job_desc_text)
        results.append({
            "candidate_id": filename,
            "resume_score": round(score, 2),
            "llm_explanation": explanation
        })

    df = pd.DataFrame(results).sort_values(by="resume_score", ascending=False)
    df.to_csv("ranked_candidates.csv", index=False)
    print("\n✅ Results saved to 'ranked_candidates.csv'")
