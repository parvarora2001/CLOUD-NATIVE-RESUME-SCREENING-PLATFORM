# main.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_resume_data(file_path):
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def preprocess_text(text_series):
    print("Preprocessing text...")
    return text_series.fillna("").str.lower()

def train_model(X, y):
    print("Training model...")
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_vec, y)
    
    return model, vectorizer

def predict(model, vectorizer, resume_texts):
    print("Scoring resumes...")
    resume_vec = vectorizer.transform(resume_texts)
    return model.predict_proba(resume_vec)[:, 1]

if __name__ == "__main__":
    # Step 1: Load dataset
    df = load_resume_data("resume_dataset.csv")  # Make sure this file exists in the same folder

    # Step 2: Preprocess resumes and target labels
    X = preprocess_text(df["resume_text"])  # Column name depends on your CSV
    y = df["fit_for_job"]  # 0 or 1 indicating poor or good fit

    # Step 3: Train model
    model, vectorizer = train_model(X, y)

    # Step 4: Score a new resume
    new_resume = ["Experienced data scientist with Python and NLP background."]
    score = predict(model, vectorizer, new_resume)

    print(f"Candidate score: {score[0]:.2f}")
