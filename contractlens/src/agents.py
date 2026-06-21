from collections import Counter
import re


def summarize_chunks(chunks):
    """Simple extractive summary from retrieved chunks."""
    if not chunks:
        return "No relevant content found."

    # Grab the first 2-3 useful sentences from the top chunk(s)
    text = " ".join(chunks[:2])
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:3]).strip()[:900]


def risk_agent(retrieved_text: str):
    """Detect a few common risky clause patterns."""
    if not retrieved_text:
        return ["No text available to analyze."]

    text = retrieved_text.lower()
    risks = []

    patterns = {
        "Automatic renewal": ["automatic renewal", "auto-renew", "renews automatically"],
        "Termination issue": ["termination for convenience", "without cause", "early termination fee"],
        "Liability risk": ["unlimited liability", "indemnify", "indemnification"],
        "One-sided change": ["may modify", "at its sole discretion", "change terms at any time"],
        "Confidentiality": ["confidential", "non-disclosure", "nda"],
    }

    for label, kws in patterns.items():
        if any(kw in text for kw in kws):
            risks.append(label)

    if not risks:
        risks.append("No obvious red-flag pattern detected in the retrieved text.")
    return risks


def answer_question(question: str, retrieved_chunks):

    if not retrieved_chunks:
        return "I could not find relevant clauses in the uploaded document."

    evidence = "\n\n".join(
        [f"[Source {i+1}] {chunk[:700]}"
         for i, (chunk, _) in enumerate(retrieved_chunks)]
    )

    summary = summarize_chunks(
        [chunk for chunk, _ in retrieved_chunks]
    )

    risks = risk_agent(evidence)

    risk_text = "\n- ".join(risks)

    response = f"""
Answer (grounded in the uploaded document):

{summary}

Key risks / flags:

- {risk_text}

Relevant excerpts:

{evidence}
"""

    return response