import streamlit as st
from src.pdf_utils import extract_text_from_pdf, chunk_text
from src.retrieval import retrieve_top_k
from src.agents import answer_question, risk_agent


st.set_page_config(page_title="ContractLens", layout="wide")

st.title("ContractLens AI: Multi-Agent Legal Intelligence System")
st.caption("Agentic RAG system for contract understanding, risk detection, grounded legal assistance, and intelligent clause retrieval.")

uploaded_file = st.file_uploader("Upload a PDF contract", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(raw_text)

    if not raw_text:
        st.error("No text could be extracted from this PDF. Try another PDF with selectable text.")
        st.stop()

    st.success(f"Loaded document with {len(chunks)} chunks.")

    col1, col2 = st.columns([2, 1])

    with col1:
        question = st.text_input(
            "Ask a question about the contract",
            placeholder="e.g. What are the termination terms? What liabilities exist?"
        )

        if st.button("Get answer") and question:
            retrieved = retrieve_top_k(question,chunks,k=3)
            answer = answer_question(question, retrieved)
            st.subheader("Answer")
            st.write(answer)

    with col2:
        st.subheader("Quick risk scan")
        top_text = " ".join(chunks[:3])
        risks = risk_agent(top_text)
        for r in risks:
            st.write(f"- {r}")

    with st.expander("View extracted text"):
        st.write(raw_text[:20000])
else:
    st.info("Upload a contract PDF to begin.")