# Multimodal RAG Agent

A Retrieval-Augmented Generation assistant that answers questions about long documents (240+ pages), including image-based content, using Gemini, LangChain, ChromaDB and FastAPI.

## Features
- Page and line citations so answers are grounded in the source text
- Conversational memory across questions
- Global summarization of whole books
- Three content modes (Coding, Novel, Manga/Comic) with Tesseract OCR for image-based pages

## Tech Stack
Python, LangChain, Gemini, ChromaDB, FastAPI, Tesseract OCR

## How to Run
1. Install Python 3.10+ and Tesseract OCR
2. cd backend and run: pip install -r requirements.txt
3. Copy .env.example to .env and add your GOOGLE_API_KEY
4. python main.py
5. Open http://localhost:8000
