# app/streamlit_app.py
# Streamlit UI for CodeDNA-RAG: index a codebase and query it with natural language

# ── Step 0: Make sure the project root is on sys.path so 'core' is importable
# when Streamlit is launched from any working directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Step 1: Imports ──────────────────────────────────────────────────────────
import streamlit as st
from core.pipeline import index_folder
from core.vector_store import query_chunks
from core.llm_handler import generate_answer


# ── Step 2: Page config (must be the very first Streamlit call) ───────────────
st.set_page_config(
    page_title="CodeDNA-RAG",
    page_icon="🧬",
    layout="wide",
)


# ── Step 3: Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import Inter font from Google Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* Global font + dark background */
  html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background-color: #0e1117;
    color: #e2e8f0;
  }

  /* Gradient hero title */
  .gradient-title {
    font-size: 3.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.3rem;
  }

  /* Muted subtitle */
  .subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
    margin-top: 0;
    margin-bottom: 1.5rem;
  }

  /* Card / panel container */
  .card {
    background: #1e2130;
    border: 1px solid #2d3348;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
    margin-bottom: 1.5rem;
  }

  /* Styled Streamlit buttons */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    font-size: 0.95rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    box-shadow: 0 2px 12px rgba(99, 102, 241, 0.35);
  }
  .stButton > button:hover {
    transform: scale(1.03);
    filter: brightness(1.12);
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.55);
  }
  .stButton > button:active {
    transform: scale(0.98);
  }

  /* Text input styling */
  .stTextInput > div > div > input {
    background-color: #1e2130;
    border: 1px solid #3a4060;
    border-radius: 8px;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
  }
  .stTextInput > div > div > input:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25);
  }

  /* Horizontal divider */
  hr {
    border: none;
    border-top: 1px solid #2d3348;
    margin: 1.2rem 0 1.8rem 0;
  }

  /* Success message */
  .stAlert {
    border-radius: 8px;
  }

  /* Answer card — slightly lighter than page bg, muted purple border */
  .answer-card {
    background: #161b2e;
    border: 1px solid #4c3f8a;
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 16px rgba(99, 102, 241, 0.12);
    color: #e2e8f0;
    font-size: 0.97rem;
    line-height: 1.7;
  }

  /* Expander header styling */
  .streamlit-expanderHeader {
    font-weight: 600;
    color: #a5b4fc;
  }

  /* Footer */
  .footer {
    border-top: 1px solid #2d3348;
    margin-top: 3rem;
    padding-top: 1rem;
    color: #4a5568;
    font-size: 0.82rem;
    text-align: center;
  }
</style>
""", unsafe_allow_html=True)


# ── Step 4: Initialise session state ─────────────────────────────────────────
if "indexed" not in st.session_state:
    st.session_state["indexed"] = False
if "chunk_count" not in st.session_state:
    st.session_state["chunk_count"] = 0


# ── Step 5: Hero section ──────────────────────────────────────────────────────
st.markdown('<p class="gradient-title">🧬 CodeDNA-RAG</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">'
    "Chat with any codebase using AST-based semantic search — "
    "not naive character splitting."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# Two columns: spacer on left, Reset button aligned right
col_spacer, col_reset = st.columns([8, 1])
with col_reset:
    # Reset button clears all session state so the user can start fresh
    if st.button("🔄 Reset", help="Clear the index and start over with a different codebase"):
        st.session_state.clear()
        st.rerun()


# ── Step 6: Index section ─────────────────────────────────────────────────────
st.markdown("### 📁 Index a Codebase")

folder_path = st.text_input(
    label="Codebase folder path",
    value="core",
    help="Absolute or relative path to the Python project you want to index.",
)

if st.button("🔍 Index Codebase"):
    if not folder_path.strip():
        st.warning("Please enter a valid folder path.")
    else:
        with st.spinner("Parsing AST, generating embeddings, and building vector index..."):
            chunk_count = index_folder(folder_path.strip())

        # Persist indexing state across reruns
        st.session_state["indexed"]        = True
        st.session_state["chunk_count"]    = chunk_count
        st.session_state["indexed_folder"] = folder_path.strip()

# Show success banner if indexing was completed (persists across reruns)
if st.session_state["indexed"]:
    st.success(
        f"✅ Indexed **{st.session_state['chunk_count']} chunks** "
        f"from `{folder_path}` — ready to query!"
    )


# ── Step 7: Chat / Q&A section (only visible after indexing) ─────────────────
if st.session_state.get("indexed"):

    st.markdown("---")
    st.markdown("### 💬 Ask a Question")

    # Wrap input + button in a form so pressing Enter also triggers the search
    with st.form(key="qa_form"):
        # Question text input — Enter key submits the form
        question = st.text_input(
            label="Your question",
            placeholder="e.g. how does session handling work",
        )

        # Submit button inside the form
        submitted = st.form_submit_button("✨ Get Answer")

    # Caption showing which folder is currently indexed
    st.caption(f"Currently indexed: {st.session_state.get('indexed_folder', folder_path)}")

    # Handle form submission
    if submitted:
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Retrieving relevant code and generating answer..."):
                # Step 7a: Retrieve the top 5 most relevant chunks from ChromaDB
                chunks = query_chunks(question)

                # Step 7b: Send chunks + question to the LLM and get an answer
                answer = generate_answer(question, chunks)

            # Step 7c: Display the answer in a styled card
            st.markdown(
                f'<div class="answer-card">{answer}</div>',
                unsafe_allow_html=True,
            )

            # Step 7d: Show retrieved chunks in a collapsible expander
            with st.expander("📄 View retrieved code chunks"):
                for i, chunk in enumerate(chunks, start=1):
                    name     = chunk["metadata"]["name"]
                    filename = chunk["metadata"]["filename"]
                    code     = chunk["document"]

                    # Chunk header: number, name, and source file
                    st.markdown(
                        f"**Chunk {i}: `{name}`** — *{filename}*"
                    )

                    # Syntax-highlighted source code
                    st.code(code, language="python")

                    # Divider between chunks (skip after the last one)
                    if i < len(chunks):
                        st.markdown("---")


# ── Step 8: Footer ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "Built by Tejas — CodeDNA-RAG uses AST-based chunking instead of "
    "naive character splitting for accurate code retrieval"
    "</div>",
    unsafe_allow_html=True,
)
