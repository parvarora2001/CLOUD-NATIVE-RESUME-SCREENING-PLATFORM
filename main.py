# main.py (Updated)
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from resume_scoring import compute_cosine_score, extract_text
from llm_explainer import explain_fit
from audio_transcription import transcribe_audio, analyze_interview_fit
import logging
import tempfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume + Interview Scoring API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

def final_decision(cosine_score, llm_score, interview_score=None):
    """Enhanced decision making with interview consideration"""
    if interview_score is None:
        # Original logic for resume-only
        if llm_score is None:
            return "Good fit" if cosine_score >= 0.3 else "Poor fit"
        if cosine_score >= 0.4 and llm_score >= 0.65:
            return "Good fit"
        elif cosine_score < 0.4 and llm_score < 0.6:
            return "Poor fit"
        elif llm_score >= 0.8:
            return "Potential fit based on context (LLM)"
        else:
            return "Poor fit"
    else:
        # Enhanced logic with interview
        combined_score = (cosine_score * 0.3 + llm_score * 0.3 + interview_score * 0.4)
        
        if combined_score >= 0.75:
            return "Excellent fit"
        elif combined_score >= 0.6:
            return "Good fit"
        elif combined_score >= 0.45:
            return "Average fit"
        else:
            return "Poor fit"

@app.get("/")
async def root():
    return {"message": "Resume + Interview Scoring API is running"}

@app.post("/score")
@app.post("/score/")
async def score_resume_only(resume: UploadFile = File(...), job: UploadFile = File(...)):
    """Original resume-only scoring"""
    try:
        logger.info(f"Resume-only analysis - Processing: {resume.filename}, {job.filename}")
        
        resume_bytes = await resume.read()
        job_bytes = await job.read()
        
        resume_text = extract_text(resume_bytes, resume.filename)
        job_text = extract_text(job_bytes, job.filename)
        
        if not resume_text.strip() or not job_text.strip():
            raise HTTPException(status_code=400, detail="Resume or job description is empty")
        
        cosine_score = compute_cosine_score(resume_text, job_text)
        explanation, llm_score = explain_fit(resume_text, job_text)
        decision = final_decision(cosine_score, llm_score)
        
        return {
            "cosine_score": round(cosine_score, 2),
            "llm_score": round(llm_score, 2) if llm_score is not None else None,
            "interview_score": None,
            "combined_score": None,
            "decision": decision,
            "explanation": explanation,
            "interview_transcript": None,
            "analysis_type": "resume_only"
        }
    except Exception as e:
        logger.error(f"Error in resume-only analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-complete")
@app.post("/analyze-complete/")
async def analyze_complete(
    resume: UploadFile = File(...), 
    job: UploadFile = File(...),
    interview: UploadFile = File(...)
):
    """Complete analysis with resume, job, and interview"""
    try:
        logger.info(f"Complete analysis - Resume: {resume.filename}, Job: {job.filename}, Interview: {interview.filename}")
        
        # Process resume and job
        resume_bytes = await resume.read()
        job_bytes = await job.read()
        interview_bytes = await interview.read()
        
        resume_text = extract_text(resume_bytes, resume.filename)
        job_text = extract_text(job_bytes, job.filename)
        
        # Process interview audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            temp_audio.write(interview_bytes)
            temp_audio_path = temp_audio.name
        
        try:
            # Transcribe interview
            interview_transcript = transcribe_audio(temp_audio_path)
            logger.info(f"Interview transcribed: {len(interview_transcript)} characters")
            
            # Analyze all components
            cosine_score = compute_cosine_score(resume_text, job_text)
            explanation, llm_score = explain_fit(resume_text, job_text)
            interview_analysis, interview_score = analyze_interview_fit(
                interview_transcript, job_text, resume_text
            )
            
            # Combined decision
            decision = final_decision(cosine_score, llm_score, interview_score)
            combined_score = (cosine_score * 0.3 + llm_score * 0.3 + interview_score * 0.4)
            
            return {
                "cosine_score": round(cosine_score, 2),
                "llm_score": round(llm_score, 2) if llm_score is not None else None,
                "interview_score": round(interview_score, 2),
                "combined_score": round(combined_score, 2),
                "decision": decision,
                "explanation": explanation,
                "interview_transcript": interview_transcript,
                "interview_analysis": interview_analysis,
                "analysis_type": "complete"
            }
            
        finally:
            # Cleanup temp file
            os.unlink(temp_audio_path)
            
    except Exception as e:
        logger.error(f"Error in complete analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)