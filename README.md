# SHL Assessment Recommendation System

This project is a web-based system that provides SHL assessment recommendations based on job descriptions. It consists of a FastAPI backend for processing requests and a Streamlit frontend for user interaction.

## Live Demo

- Frontend (Streamlit): http://localhost:8501
- Backend API (FastAPI): http://localhost:8000

## Features

- Job description analysis
- SHL assessment recommendations
- User-friendly web interface
- RESTful API endpoints

## Technology Stack

- Backend: FastAPI
- Frontend: Streamlit
- ML: scikit-learn, sentence-transformers
- Python: 3.11.0

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the FastAPI backend:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Start the Streamlit frontend:
   ```bash
   streamlit run streamlit_app.py
   ```

## API Documentation

Access the API documentation at `/docs` endpoint when the backend server is running.

## Deployment

The application is configured for cloud deployment with the following files:
- Procfile: Contains commands for running both backend and frontend
- runtime.txt: Specifies Python version
- requirements.txt: Lists all dependencies

## License

MIT License