import streamlit as st #type: ignore
from sentence_transformers import SentenceTransformer, util
from docx import Document
import fitz
from llm_explainer import explain_fit  

MODEL_NAME = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(MODEL_NAME)

def read_docx(uploaded_file):
    doc = Document(uploaded_file)
    return '\n'.join(para.text for para in doc.paragraphs)

def read_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ''
    for page in doc:
        text += page.get_text()
    return text

def read_resume_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        return read_pdf(uploaded_file)
    else:
        st.error(f"Unsupported file type: {uploaded_file.type}")
        return None

def score_resume_text(resume_text, job_desc_text):
    job_embedding = embedder.encode(job_desc_text, convert_to_tensor=True)
    resume_embedding = embedder.encode(resume_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(job_embedding, resume_embedding).item()
    return similarity

# Your LLM explain_fit function goes here, e.g.
# from llm_ollama import query_llm
# from llm_explainer import explain_fit

st.title("Resume Screening & Explanation")

job_desc_text = st.text_area("Paste Job Description here", height=200)

uploaded_files = st.file_uploader(
    "Upload multiple resumes (.txt, .docx, .pdf)",
    type=["txt", "docx", "pdf"],
    accept_multiple_files=True,
)

explain_top_n = st.number_input("Number of top candidates to explain", min_value=1, max_value=10, value=3)

if st.button("Score & Explain"):

    if not job_desc_text.strip():
        st.error("Please enter the job description text.")
    elif not uploaded_files or len(uploaded_files) == 0:
        st.error("Please upload at least one resume file.")
    else:
        resumes = {}
        for file in uploaded_files:
            text = read_resume_file(file)
            if text:
                resumes[file.name] = text

        if len(resumes) == 0:
            st.error("No valid resumes to process.")
        else:
            with st.spinner("Scoring resumes..."):
                scores = {}
                for name, text in resumes.items():
                    scores[name] = score_resume_text(text, job_desc_text)

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            st.subheader("Ranked Resumes")
            for rank, (name, score) in enumerate(ranked, start=1):
                st.write(f"**{rank}. {name} — Score: {score:.2f}**")

            st.subheader(f"Explanations for Top {explain_top_n} Candidates")
            for name, score in ranked[:explain_top_n]:
                st.write(f"### {name} (Score: {score:.2f})")
                explanation = explain_fit(resumes[name], job_desc_text)
                st.write(explanation)
