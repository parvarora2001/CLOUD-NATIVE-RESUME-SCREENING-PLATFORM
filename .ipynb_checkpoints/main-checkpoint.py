from resume_scoring import score_resumes
from audio_transcription import transcribe_audio
from transcript_scoring import score_transcripts
from model_training import create_feature_df, train_model

JOB_DESC_PATH = 'data/job_description.txt'
RESUME_DIR = 'data/resumes/'
INTERVIEW_AUDIO_DIR = 'data/interviews/'
TRANSCRIPT_DIR = 'data/transcripts/'

if __name__ == "__main__":
    print("Scoring resumes...")
    resume_scores = score_resumes(JOB_DESC_PATH, RESUME_DIR)

    print("Transcribing interview audio...")
    transcribe_audio(INTERVIEW_AUDIO_DIR, TRANSCRIPT_DIR)

    print("Scoring transcripts...")
    transcript_scores = score_transcripts(JOB_DESC_PATH, TRANSCRIPT_DIR)

    print("Creating features...")
    df = create_feature_df(resume_scores, transcript_scores)

    print("Training model...")
    trained_model = train_model(df)

    df_sorted = df.sort_values(by=['resume_score', 'transcript_score'], ascending=False)
    df_sorted.to_csv('ranked_candidates.csv', index=False)
    print("Candidate ranking saved to 'ranked_candidates.csv'")
