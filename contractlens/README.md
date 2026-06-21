# ContractLens

A lightweight Legal Services RAG demo for hackathon Round 1.

## Features
- Upload a contract PDF
- Chunk and retrieve relevant clauses
- Ask questions about the document
- Detect a few common contract risk patterns

## Tech Stack
- Streamlit
- PyPDF
- TF-IDF retrieval
- Python

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py


---

## 9) Build order for today

1. Create GitHub account
2. Create a new repository named `contractlens`
3. Paste these files locally
4. Run the app on your laptop first
5. Push to GitHub
6. Then we can deploy it to Streamlit Cloud

---

## 10) Important note

This is intentionally simple so it can be finished fast. It is a valid **Round 1 MVP** because it shows:
- document ingestion
- retrieval
- grounded answer generation
- risk analysis
- clean structure for a hackathon submission