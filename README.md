# SHL Assessment Chatbot

A conversational AI agent that helps hiring managers and recruiters select the right SHL Individual Test Solutions for their hiring needs.

## Features
- **Retrieval Augmented Generation (RAG)**: Uses FAISS and `sentence-transformers` for semantic search over the SHL catalog.
- **Dual LLM Providers**: Primary Google Gemini, with Groq (Llama 3) as fallback.
- **Robust Guardrails**: Enforces conversational rules (clarify, recommend, refine, compare) and prevents hallucinated recommendations through post-validation against the ground truth catalog.
- **FastAPI**: Exposes `/health` and `/chat` endpoints conforming exactly to the expected JSON schema.

## Setup
1. Create `.env` file with `GEMINI_API_KEY` and `GROQ_API_KEY`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`

See `approach_document.md` for full implementation details.
