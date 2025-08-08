# Use a lightweight Python base image
FROM python:3.10-slim-bullseye

# Create and set working directory
WORKDIR /app

# Install system dependencies required by PyMuPDF and sentence-transformers
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip 
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
# Copy all project files into the container
COPY . .

# Set environment variables for Cloud Run
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose FastAPI default port
EXPOSE 8080

# Run the FastAPI app using uvicorn with more workers for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]