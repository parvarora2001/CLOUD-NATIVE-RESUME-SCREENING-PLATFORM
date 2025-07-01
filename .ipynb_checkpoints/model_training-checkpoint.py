import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def create_feature_df(resume_scores, transcript_scores):
    data = []
    for key in resume_scores:
        candidate_id = key.replace('.txt', '')
        resume_score = resume_scores[key]
        transcript_score = transcript_scores.get(key, 0)
        label = 1 if resume_score + transcript_score > 1.2 else 0
        data.append({
            "candidate_id": candidate_id,
            "resume_score": resume_score,
            "transcript_score": transcript_score,
            "label": label
        })
    return pd.DataFrame(data)

def train_model(df):
    X = df[["resume_score", "transcript_score"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model
