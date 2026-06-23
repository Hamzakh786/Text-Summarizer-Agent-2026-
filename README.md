# Text-Summarizer-Agent-2026-
Text Summarizer Agent using Hugging Face and FastAPI – An AI-powered text summarization service that leverages Hugging Face Transformer models and FastAPI to generate concise and meaningful summaries from lengthy text documents through a REST API.
Text Summarizer Agent

An AI-powered text summarization application built using Hugging Face Transformers and FastAPI. The system processes long-form text and generates accurate, concise summaries using state-of-the-art NLP models.

Features
Automatic text summarization
REST API using FastAPI
Hugging Face Transformer integration
Supports long-form articles and documents
JSON-based API responses
Interactive API documentation with Swagger UI
Tech Stack
Python
FastAPI
Hugging Face Transformers
PyTorch
Uvicorn


Project Structure
Text-Summarizer-Agent/
│
├── app/
│   ├── main.py
│   ├── summarizer.py
│   └── models.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── sample_input.txt
API Endpoint
POST /summarize

Request:

{
  "text": "Long article text..."
}

Response:

{
  "summary": "Generated summary text..."
}
Installation
git clone https://github.com/yourusername/Text-Summarizer-Agent.git

cd Text-Summarizer-Agent

pip install -r requirements.txt

uvicorn app.main:app --reload
