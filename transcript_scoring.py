import os
import pandas as pd
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit([text1, text2])
    tfidf = vectorizer.transform([text1, text2])
    return (tfidf[0] @ tfidf[1].T).A[0][0]

def load_transcripts(folder):
    transcripts = {}
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding='utf-8') as f:
                transcripts[file] = f.read()
    return transcripts

def match_transcript_to_candidate(transcript_file, candidate_files):
    best_match = None
    best_ratio = 0
    base_t = os.path.splitext(transcript_file)[0].lower()

    for c in candidate_files:
        base_c = os.path.splitext(c)[0].lower()
        ratio = SequenceMatcher(None, base_t, base_c).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = c

    return best_match if best_ratio > 0.5 else None

def score_transcripts(job_desc, transcript_folder, candidate_files):
    transcripts = load_transcripts(transcript_folder)
    transcript_scores = {c: 0.0 for c in candidate_files}

    for filename, text in transcripts.items():
        match = match_transcript_to_candidate(filename, candidate_files)
        if match:
            score = compute_similarity(job_desc, text)
            transcript_scores[match] = round(score, 2)
            print(f"{filename} → matched with {match} → score: {score:.2f}")
        else:
            print(f"{filename} → no match found.")

    return transcript_scores

def create_feature_df(resume_scores, transcript_scores):
    df = pd.DataFrame({
        'candidate_id': list(resume_scores.keys()),
        'resume_score': list(resume_scores.values()),
        'transcript_score': [transcript_scores.get(k, 0.0) for k in resume_scores]
    })

    # Optional: Remove this part if you don't want dummy or external labels
    df['label'] = (df['resume_score'] + df['transcript_score'] > 1.2).astype(int)
    return df

def train_model(df):
    X = df[['resume_score', 'transcript_score']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, model.predict(X_test)))
    return model
