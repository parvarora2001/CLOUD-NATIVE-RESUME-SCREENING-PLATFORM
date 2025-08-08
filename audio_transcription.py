import whisper
import tempfile
import os
from llm_explainer import explain_fit  # Reuse your existing LLM logic

def transcribe_audio(audio_path):
    """Transcribe audio file to text using Whisper"""
    try:
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        raise Exception(f"Audio transcription failed: {str(e)}")

def calculate_interview_score(analysis_text, interview_text):
    """Calculate interview score based on text analysis"""
    score = 0.5  # Base score
    
    # Positive indicators
    positive_keywords = [
        'excellent', 'strong', 'good', 'experienced', 'skilled',
        'confident', 'clear', 'detailed', 'relevant', 'appropriate'
    ]
    
    # Negative indicators  
    negative_keywords = [
        'unclear', 'vague', 'insufficient', 'weak', 'poor',
        'irrelevant', 'inconsistent', 'unprepared'
    ]
    
    analysis_lower = analysis_text.lower()
    
    # Adjust score based on keyword presence
    for word in positive_keywords:
        if word in analysis_lower:
            score += 0.05
    
    for word in negative_keywords:
        if word in analysis_lower:
            score -= 0.05
    
    # Consider interview length (longer usually better, but cap it)
    word_count = len(interview_text.split())
    if word_count > 500:  # Substantial interview
        score += 0.1
    elif word_count < 100:  # Very short interview
        score -= 0.1
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))

def analyze_interview_fit(interview_text, job_text, resume_text):
    """Analyze interview transcript for job fit"""
    
    # Create a comprehensive prompt for interview analysis
    interview_prompt = f"""
    Analyze this job interview transcript for job fit:
    
    JOB REQUIREMENTS:
    {job_text}
    
    CANDIDATE'S RESUME:
    {resume_text}
    
    INTERVIEW TRANSCRIPT:
    {interview_text}
    
    Please analyze:
    1. Communication skills and clarity
    2. Technical knowledge relevant to the job
    3. Cultural fit and enthusiasm
    4. Consistency between resume claims and interview responses
    5. Problem-solving approach demonstrated
    6. Overall interview performance
    
    Provide a detailed analysis and score from 0.0 to 1.0.
    """
    
    try:
        # Use your existing LLM explanation function with modified prompt
        analysis, score = explain_fit(interview_prompt, job_text)
        
        # If your explain_fit doesn't return a score, create one based on analysis
        if score is None:
            # Simple scoring based on key indicators in the analysis
            score = calculate_interview_score(analysis, interview_text)
            
        return analysis, score
        
    except Exception as e:
        return f"Interview analysis failed: {str(e)}", 0.0

