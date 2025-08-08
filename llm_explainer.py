import requests
import re

GEMMA_API_URL = "https://ollama-gemma-757300761760.us-central1.run.app"

def explain_fit(resume, job):
    prompt = (
        f"You are an expert hiring manager at a reputed company with decades of experience in hiring people. "
        f"I will provide you two texts: a job description and a candidate's resume that you have to evaluate based on the job description.\n\n"
        f"Please perform the following tasks:\n"
        f"1. Assess how well the candidate's resume fits the job description based on relevant skills, work experience or the clients they worked for as a freelancer, and context.\n"
        f"2. Give a numeric fit score between 0 and 1, where 0 means no fit and 1 means perfect fit. Be strict and realistic: only assign high scores if the resume clearly matches the job requirements.\n"
        f"3. Provide a brief explanation (1-2 sentences) justifying the score, highlighting key matches or mismatches.\n\n"
        f"Job Description:\n{job}\n\nResume:\n{resume}\n\n"
        f"Please provide your response in the format: 'Score: X.XX' followed by your explanation."
    )
    
    try:
        response = requests.post(
            GEMMA_API_URL + "/api/generate",  # Common Ollama endpoint
            json={
                "model": "gemma2",  # Adjust model name as needed
                "prompt": prompt,
                "stream": False
            },
            timeout=60  # Add timeout for Cloud Run
        )
        
        if response.status_code == 200:
            explanation = response.json().get("response", "")
            score = parse_llm_score(explanation)
            return explanation, score
        else:
            print(f"LLM API error: {response.status_code}")
            return "LLM service unavailable", None
            
    except requests.exceptions.RequestException as e:
        print(f"LLM request failed: {e}")
        return "LLM service unavailable", None

def parse_llm_score(text):
    # Look for patterns like "Score: 0.85" or just "0.85"
    patterns = [
        r'[Ss]core:\s*(\d?\.\d+)',
        r'(\d\.\d+)',
        r'(\d)/10',  # Handle X/10 format
        r'(\d+)%'    # Handle percentage format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                score = float(match.group(1))
                # Normalize score to 0-1 range
                if score > 1:
                    score = score / 10 if score <= 10 else score / 100
                return score if 0 <= score <= 1 else None
            except:
                continue
    return None