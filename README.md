# ContractLens AI: Multi-Agent Legal Intelligence System

## Problem Statement

Legal teams and consultants spend significant time reviewing lengthy contracts, identifying risks, and searching for relevant clauses manually.

## Solution

ContractLens AI is an agentic RAG (Retrieval-Augmented Generation) system that lets users upload a contract PDF, ask questions in plain English, and get grounded answers, plain-language summaries, and risk flags — powered by real semantic search and Google's Gemini LLM.

## How it actually works

1. **PDF Parsing** — `pypdf` extracts raw text from the uploaded contract.
2. **Chunking** — text is split into ~1200-character overlapping chunks so clauses aren't cut off mid-sentence.
3. **Embedding & Retrieval** — every chunk is embedded using Gemini's `text-embedding-004` model. A user's question is embedded the same way, and the chunks whose *meaning* is closest (via cosine similarity) are retrieved — not just chunks sharing literal keywords.
4. **Orchestrator Agent** — coordinates the specialist agents below for each question, using the same retrieved evidence for all of them so answers stay grounded and consistent.
5. **Specialist Agents (each a distinct Gemini prompt):**
   - **Summarizer Agent** — plain-language summary of the retrieved clauses.
   - **Risk Agent** — flags risky or one-sided clauses (auto-renewal, unlimited liability, unilateral termination, etc.) using LLM reasoning rather than hardcoded keyword lists.
   - **QA Agent** — answers the user's specific question, grounded strictly in the retrieved excerpts.

## Tech Stack

- Python
- Streamlit (UI)
- pypdf (PDF text extraction)
- Google Gemini API (`google-genai`) — embeddings (`gemini-embedding-001`) and generation (`gemini-flash-latest`, Google's auto-updating alias for their current Flash model)
- NumPy (cosine similarity)

## Architecture

```
PDF Upload
    ↓
PDF Parsing
    ↓
Chunk Generation
    ↓
Embedding (Gemini gemini-embedding-001)
    ↓
Retrieval Engine (cosine similarity)
    ↓
Orchestrator Agent
    ↓
    ├── Summarizer Agent (Gemini)
    ├── Risk Agent (Gemini)
    └── QA Agent (Gemini)
    ↓
Grounded Response
```

## Setup & Run Locally

```bash
pip install -r requirements.txt

# Get a free API key: https://aistudio.google.com/apikey
export GEMINI_API_KEY="your-key-here"   # or paste it in the sidebar when the app runs

python -m streamlit run app.py
```

## Known limitations / future scope

- Retrieval currently compares each new question against in-memory embeddings recomputed per session; a persistent vector database (e.g. FAISS or Chroma) would let the app scale to many documents without re-embedding.
- Risk detection depends on the LLM's judgment rather than a verified legal-review process — outputs should be treated as an assistive first pass, not legal advice.
