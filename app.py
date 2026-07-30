import streamlit as st
from src.pdf_utils import extract_text_from_pdf, chunk_text
from src.embeddings import get_client, embed_texts
from src.orchestrator import handle_question, handle_quick_scan


st.set_page_config(page_title="ContractLens AI", page_icon="⚖️", layout="wide")

# --- Minimal CSS injection for a clean, professional legal-tech look ---
# This is plain CSS layered on top of Streamlit's default components —
# no new frontend framework, just styling the existing HTML Streamlit
# already renders under the hood.
st.markdown("""
<style>
:root {
    --navy: #1F3B57;
    --navy-light: #2C5578;
    --paper: #FAFAF8;
}
.stApp { background-color: var(--paper); }

/* Header banner */
.cl-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
    padding: 1.6rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 14px rgba(31,59,87,0.18);
}
.cl-header h1 {
    color: white;
    font-size: 1.9rem;
    margin: 0;
    font-weight: 700;
}
.cl-header p {
    color: #DCE6EF;
    margin: 0.3rem 0 0 0;
    font-size: 0.95rem;
}

/* Card containers for results */
.cl-card {
    background: white;
    border: 1px solid #E3E3DE;
    border-left: 4px solid var(--navy);
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.cl-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(31,59,87,0.12);
}
.cl-card h4 {
    color: var(--navy);
    margin-top: 0;
    margin-bottom: 0.5rem;
    font-size: 1.05rem;
}
.cl-risk-card { border-left-color: #B3541E; }
.cl-risk-card h4 { color: #B3541E; }

/* Buttons */
.stButton>button, .stFormSubmitButton>button {
    background-color: var(--navy);
    color: white;
    border-radius: 6px;
    border: none;
    font-weight: 600;
}
.stButton>button:hover, .stFormSubmitButton>button:hover {
    background-color: var(--navy-light);
    color: white;
}
/* Metric tiles */
div[data-testid="stMetric"] {
    background: white;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    border: 1px solid #E3E3DE;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
div[data-testid="stMetricValue"] { color: var(--navy); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cl-header">
    <h1>⚖️ ContractLens AI</h1>
    <p>Multi-Agent Legal Intelligence System — Agentic RAG for contract understanding, risk detection, and grounded Q&A</p>
</div>
""", unsafe_allow_html=True)

# --- API key handling ---
# We never hardcode the key in the file. It's read from Streamlit secrets
# (for deployed apps) or an environment variable (for local runs), or the
# user can paste it in the sidebar for a quick local test.
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError, StopIteration, Exception):
    api_key = None
if not api_key:
    st.sidebar.markdown("### 🔑 Setup")
    api_key = st.sidebar.text_input("Gemini API key", type="password",
                                     help="Get a free key at https://aistudio.google.com/apikey")

if not api_key:
    st.warning("Enter your Gemini API key in the sidebar to use ContractLens.")
    st.stop()

client = get_client(api_key)

uploaded_file = st.file_uploader("📄 Upload a PDF contract", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(raw_text)

    if not raw_text:
        st.error("No text could be extracted from this PDF. Try another PDF with selectable text.")
        st.stop()

    st.success(f"✅ Loaded document with {len(chunks)} chunks.")

    m1, m2, m3 = st.columns(3)
    m1.metric("📄 Document", uploaded_file.name[:20] + ("…" if len(uploaded_file.name) > 20 else ""))
    m2.metric("🧩 Chunks", len(chunks))
    m3.metric("❓ Questions asked", st.session_state.get("question_count", 0))

    # Embed all chunks once per uploaded document, cache in session_state
    # so we don't re-embed on every interaction (saves API calls + time).
    cache_key = f"embeddings_{uploaded_file.name}_{len(chunks)}"
    if cache_key not in st.session_state:
        with st.spinner("Embedding document (one-time per upload)..."):
            st.session_state[cache_key] = embed_texts(client, chunks)
    chunk_embeddings = st.session_state[cache_key]

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.form(key="question_form"):
            question = st.text_input(
                "Ask a question about the contract",
                placeholder="e.g. What are the termination terms? What liabilities exist?"
            )
            submitted = st.form_submit_button("Get answer")

        if submitted and question:
            with st.spinner("Thinking..."):
                try:
                    result = handle_question(client, question, chunks, chunk_embeddings)
                except Exception as e:
                    st.error(f"⚠️ Couldn't get a response ({type(e).__name__}). "
                             f"If this says quota/rate limit, the free API tier caps requests per day/minute — wait a bit and retry.")
                    st.stop()

            st.session_state["question_count"] = st.session_state.get("question_count", 0) + 1

            st.markdown(f"""
            <div class="cl-card">
                <h4>💬 Answer</h4>
                <p>{result["answer"]}</p>
            </div>
            """, unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["📝 Summary", "⚠️ Risk Findings", "📚 Source Excerpts"])
            with tab1:
                st.markdown(f"""
                <div class="cl-card"><p>{result["summary"]}</p></div>
                """, unsafe_allow_html=True)
            with tab2:
                st.markdown(f"""
                <div class="cl-card cl-risk-card"><p>{result["risks"]}</p></div>
                """, unsafe_allow_html=True)
            with tab3:
                for i, src in enumerate(result["sources"]):
                    st.markdown(f"""
                    <div class="cl-card"><h4>Excerpt {i+1}</h4><p>{src}</p></div>
                    """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🔍 Quick Risk Scan")
        scan_cache_key = f"quickscan_{uploaded_file.name}_{len(chunks)}"
        if scan_cache_key not in st.session_state:
            with st.spinner("Scanning..."):
                try:
                    st.session_state[scan_cache_key] = handle_quick_scan(client, chunks, chunk_embeddings)
                except Exception as e:
                    st.session_state[scan_cache_key] = f"⚠️ Could not complete scan right now ({type(e).__name__}). Try again in a moment — the free API tier has a daily/per-minute request limit."
        risks = st.session_state[scan_cache_key]
        st.markdown(f"""
        <div class="cl-card cl-risk-card"><p>{risks}</p></div>
        """, unsafe_allow_html=True)

    with st.expander("📄 View extracted text"):
        st.write(raw_text[:20000])
else:
    st.info("👆 Upload a contract PDF to begin.")
