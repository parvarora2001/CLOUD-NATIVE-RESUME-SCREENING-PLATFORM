# CLOUD-NATIVE-RESUME-SCREENING-PLATFORM

# 🎯 Cloud-Native Resume Screening Platform

An AI-powered hiring assistant that evaluates candidates using **resume-to-job-description matching** and optionally **interview audio analysis**. Built with FastAPI, deployed on **Google Cloud Run**, and served via **Firebase Hosting**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Firebase Hosting (Frontend)                 │
│                  public/index.html                       │
│         Resume Only ◄──────────► Complete Analysis      │
└────────────────────┬────────────────────────────────────┘
                     │ HTTPS
          ┌──────────▼──────────────────┐
          │   Google Cloud Run (API)     │
          │   FastAPI · Port 8080        │
          │                             │
          │  POST /score                │  ← resume + job description
          │  POST /analyze-complete     │  ← resume + job + audio
          └──────┬────────────┬─────────┘
                 │            │
     ┌───────────▼──┐  ┌──────▼──────────────────┐
     │  Sentence     │  │  Ollama / Gemma2         │
     │  Transformers │  │  (Cloud Run sidecar)     │
     │  all-mpnet    │  │  ollama-gemma-*.run.app  │
     │  Cosine sim   │  │  LLM scoring + reasoning │
     └───────────────┘  └──────────────────────────┘
                 │
     ┌───────────▼──────────────────┐
     │  OpenAI Whisper (small)      │
     │  Interview audio → text      │
     └──────────────────────────────┘
```

---

## Features

**Resume-Only Mode (`POST /score`)**
- Extracts text from `.pdf`, `.docx`, or `.txt` resumes
- Computes semantic similarity via `sentence-transformers` (cosine score)
- Sends resume + job description to Gemma2 LLM for expert analysis and fit score
- Returns a structured decision: `Good fit` / `Poor fit` / `Potential fit based on context`

**Complete Analysis Mode (`POST /analyze-complete`)**
- Everything above, plus:
- Transcribes interview audio (`.mp3`, `.wav`, `.m4a`) using **OpenAI Whisper**
- Evaluates interview performance against job requirements via LLM
- Combines all three signals into a weighted final score

---

## Scoring System

### Resume-Only
| Signal | Weight | Source |
|---|---|---|
| Cosine Score | Threshold check | `sentence-transformers` |
| LLM Score | Primary signal | Gemma2 via Ollama |

### Complete Analysis
| Signal | Weight | Source |
|---|---|---|
| Cosine Score | 30% | `sentence-transformers` |
| LLM Score | 30% | Gemma2 resume fit |
| Interview Score | 40% | Gemma2 interview analysis |

### Decision Thresholds
| Combined Score | Decision |
|---|---|
| ≥ 0.75 | ✅ Excellent fit |
| ≥ 0.60 | ✅ Good fit |
| ≥ 0.45 | ⚠️ Average fit |
| < 0.45 | ❌ Poor fit |

---

## Project Structure

```
CLOUD-NATIVE-RESUME-SCREENING-PLATFORM/
├── main.py                  # FastAPI app — /score and /analyze-complete endpoints
├── resume_scoring.py        # SentenceTransformer cosine similarity + text extraction
├── llm_explainer.py         # Gemma2 (Ollama) API client — score parsing + explanation
├── audio_transcription.py   # Whisper transcription + interview fit analysis
├── requirements.txt         # Python dependencies
├── Dockerfile               # python:3.10-slim-bullseye, port 8080
├── .dockerignore
├── .gcloudignore
├── firebase.json            # Firebase Hosting config (rewrites → index.html)
├── .firebaserc              # Firebase project: resume-screening-fb491
├── public/
│   ├── index.html           # Frontend SPA — file upload UI + results display
│   └── 404.html
└── data/
    ├── job_description.txt  # Sample job description (Senior Business Analyst)
    ├── resumes/             # Sample resumes (.docx)
    ├── interviews/          # Sample interview audio (.mp3)
    └── transcripts/         # Sample transcripts (.txt)
```

---

## Prerequisites

- Python 3.10+
- Docker
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)
- [Firebase CLI](https://firebase.google.com/docs/cli) (`npm install -g firebase-tools`)
- A GCP project with Cloud Run and Artifact Registry enabled
- An Ollama + Gemma2 instance deployed on Cloud Run (see [LLM Service](#llm-service) below)

---

## Local Development

### 1. Clone and install dependencies

```bash
git clone https://github.com/parvarora2001/CLOUD-NATIVE-RESUME-SCREENING-PLATFORM.git
cd CLOUD-NATIVE-RESUME-SCREENING-PLATFORM

pip install -r requirements.txt
```

### 2. Run the API locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

API will be available at `http://localhost:8080`
Interactive docs at `http://localhost:8080/docs`

### 3. Test with sample data

```bash
# Resume-only scoring
curl -X POST http://localhost:8080/score \
  -F "resume=@data/resumes/Robinson.docx" \
  -F "job=@data/job_description.txt"

# Complete analysis (resume + interview audio)
curl -X POST http://localhost:8080/analyze-complete \
  -F "resume=@data/resumes/Robinson.docx" \
  -F "job=@data/job_description.txt" \
  -F "interview=@data/interviews/Tooran-interview.mp3"
```

---

## Docker

### Build

```bash
docker build -t resume-screening-api .
```

> **Note:** The build pre-downloads the `all-mpnet-base-v2` model into the image layer. First build takes ~5–10 minutes depending on your connection.

### Run locally

```bash
docker run -p 8080:8080 resume-screening-api
```

---

## Cloud Deployment

### Deploy API to Google Cloud Run

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID

# Build and push image
gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/resume-screening-api

# Deploy to Cloud Run
gcloud run deploy resume-screening-api \
  --image gcr.io/YOUR_GCP_PROJECT_ID/resume-screening-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300
```

> ⚠️ Set **minimum memory to 4Gi**. Whisper (`small` model) + SentenceTransformers together require ~3GB RAM at inference time. Under-provisioning causes silent OOM crashes.

### Deploy Frontend to Firebase Hosting

```bash
# Login to Firebase
firebase login

# Deploy
firebase deploy --only hosting
```

The frontend will be live at `https://resume-screening-fb491.web.app`

---

## LLM Service

The platform calls an **Ollama + Gemma2** service running as a separate Cloud Run deployment at:

```
https://ollama-gemma-757300761760.us-central1.run.app
```

To deploy your own Ollama instance on Cloud Run, see the [Ollama Cloud Run guide](https://cloud.google.com/run/docs/tutorials/gpu-gemma-with-ollama). Update `GEMMA_API_URL` in `llm_explainer.py` to point to your instance.

---

## API Reference

### `GET /`
Health check.

**Response:**
```json
{ "message": "Resume + Interview Scoring API is running" }
```

---

### `POST /score`
Resume-only candidate evaluation.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|---|---|---|---|
| `resume` | file | ✅ | Candidate resume (`.pdf`, `.docx`, `.txt`) |
| `job` | file | ✅ | Job description (`.pdf`, `.docx`, `.txt`) |

**Response:**
```json
{
  "cosine_score": 0.72,
  "llm_score": 0.81,
  "interview_score": null,
  "combined_score": null,
  "decision": "Good fit",
  "explanation": "The candidate demonstrates strong alignment with the required ML and Python skills...",
  "interview_transcript": null,
  "analysis_type": "resume_only"
}
```

---

### `POST /analyze-complete`
Full evaluation with resume, job description, and interview audio.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|---|---|---|---|
| `resume` | file | ✅ | Candidate resume (`.pdf`, `.docx`, `.txt`) |
| `job` | file | ✅ | Job description (`.pdf`, `.docx`, `.txt`) |
| `interview` | file | ✅ | Interview recording (`.mp3`, `.wav`, `.m4a`) |

**Response:**
```json
{
  "cosine_score": 0.68,
  "llm_score": 0.74,
  "interview_score": 0.82,
  "combined_score": 0.75,
  "decision": "Excellent fit",
  "explanation": "Strong resume match with relevant ML experience...",
  "interview_transcript": "Interviewer: Tell me about your experience with Python...",
  "interview_analysis": "Candidate demonstrated clear communication and strong technical depth...",
  "analysis_type": "complete"
}
```

> ⚠️ Interview transcription via Whisper can take **2–4 minutes** for a typical 30-minute recording. The frontend sets a 5-minute timeout.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI 0.x + Uvicorn |
| Semantic Similarity | `sentence-transformers` · `all-mpnet-base-v2` |
| LLM Reasoning | Gemma2 via Ollama (self-hosted on Cloud Run) |
| Audio Transcription | OpenAI Whisper (`small` model) |
| PDF Parsing | PyMuPDF (`fitz`) |
| DOCX Parsing | `python-docx` |
| Containerisation | Docker · `python:3.10-slim-bullseye` |
| Backend Hosting | Google Cloud Run |
| Frontend Hosting | Firebase Hosting |
| Frontend | Vanilla HTML/CSS/JS (no build step) |

---

## Known Limitations

- **Cold starts:** Cloud Run scales to zero between requests. First request after idle can take 15–30 seconds while the container loads Whisper + SentenceTransformers into memory.
- **Interview scoring:** Keyword-based heuristics are used as a fallback when the LLM fails to return a parseable score. Not a substitute for LLM output quality.
- **Supported resume formats:** `.pdf`, `.docx`, `.txt` only. `.doc` (legacy Word) is not supported.
- **Audio formats:** `.mp3`, `.wav`, `.m4a` only.
- **LLM dependency:** If the Ollama/Gemma2 Cloud Run service is unavailable, the API returns `"LLM service unavailable"` and falls back to cosine-score-only decisions.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.
