# llm_explainer.py
from llm_ollama import query_llm

def explain_fit(resume_text, job_desc_text):
    prompt = f"""
You are an expert recruiter.

--- Job Description ---
{job_desc_text}

--- Resume ---
{resume_text}

Explain which parts of the resume match the job description. Highlight relevant skills, experience, or projects.
"""
    return query_llm(prompt).strip()
