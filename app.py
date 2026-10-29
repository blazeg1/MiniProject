
import streamlit as st
import re
import hashlib
from pathlib import Path
import pymupdf
from rank_bm25 import BM25Okapi
from common import model, collection

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="StudyVault AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #172554 0%, transparent 35%),
        #0b1020;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #10182b;
    border-right: 1px solid #26334d;
}

.hero {
    padding: 34px;
    border-radius: 24px;
    background: linear-gradient(120deg, #172554, #312e81, #164e63);
    border: 1px solid #40518a;
    margin-bottom: 26px;
}

.hero h1 {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff;
    margin: 8px 0;
}

.hero p {
    color: #dbeafe;
    font-size: 1rem;
    margin-bottom: 0;
}

.eyebrow {
    color: #93c5fd;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.metric-card {
    background: linear-gradient(145deg, #151f35, #111827);
    border: 1px solid #2a3956;
    border-radius: 18px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #9caec9;
    font-size: 0.85rem;
}

.metric-value {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 8px;
}

.section-title {
    color: #f8fafc;
    font-size: 1.35rem;
    font-weight: 700;
    margin: 24px 0 12px;
}

div[data-testid="stExpander"] {
    background: #111a2c;
    border: 1px solid #2a3956;
    border-radius: 14px;
    margin-bottom: 10px;
}

div[data-testid="stFileUploader"] {
    background: #111a2c;
    border: 1px dashed #526b9c;
    padding: 12px;
    border-radius: 16px;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #93c5fd;
    transform: translateY(-1px);
}

div[data-testid="stTextInput"] input {
    border-radius: 12px;
}

.footer {
    color: #8291ad;
    text-align: center;
    padding: 25px 0 5px;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HELPERS ----------------
DATA_DIR = Path("Data") / "uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def index_pdf(uploaded_file):
    """Save and index only a newly uploaded PDF."""
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = collection.get(
        where={"source_hash": file_hash},
        limit=1,
        include=["metadatas"]
    )

    if existing["ids"]:
        return "duplicate", 0

    safe_name = Path(uploaded_file.name).name
    saved_path = DATA_DIR / f"{file_hash[:12]}_{safe_name}"

    documents = []
    metadatas = []
    ids = []

    saved_path.write_bytes(file_bytes)

    try:
        with pymupdf.open(saved_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text().strip()

                if not text:
                    continue

                for chunk_num, start in enumerate(range(0, len(text), 800)):
                    chunk = text[start:start + 800]

                    documents.append(chunk)
                    metadatas.append({
                        "filename": safe_name,
                        "page": page_num,
                        "source_hash": file_hash
                    })
                    ids.append(
                        f"{file_hash}_{page_num}_{chunk_num}"
                    )

        if not documents:
            saved_path.unlink(missing_ok=True)
            return "no_text", 0

        embeddings = model.encode(
            documents,
            batch_size=64,
            show_progress_bar=False
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return "success", len(documents)

    except Exception:
        saved_path.unlink(missing_ok=True)
        raise


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 📚 StudyVault")
    st.caption("Your personal AI study space")
    st.divider()

    st.markdown("### 🧭 Navigation")
    page = st.radio(
    "Choose a section",
    [
        "🔎 Smart Search",
        "📚 Document Library",
        "📤 Upload Documents",
        "ℹ️ About"
    ],
    label_visibility="collapsed"
)

    st.divider()
    st.markdown("### ⚡ Quick tips")
    st.caption("• Ask a complete question")
    st.caption("• Include important technical terms")
    st.caption("• Open results to view source passages")

    st.divider()
    st.caption("Local semantic search")
    st.caption("StudyVault AI · v1.0")


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">YOUR KNOWLEDGE, CONNECTED</div>
    <h1>Study smarter with AI. ✨</h1>
    <p>Search your academic PDFs by meaning, discover relevant pages,
    and find the information you need faster.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- DASHBOARD METRICS ----------------
total_chunks = collection.count()

st.markdown('<div class="section-title">Your workspace</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📄 Indexed passages</div>
        <div class="metric-value">{total_chunks:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🧠 Search engine</div>
        <div class="metric-value">Semantic AI</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">📍 Source references</div>
        <div class="metric-value">Page-level</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- DOCUMENT LIBRARY ----------------
if page == "📚 Document Library":
    st.markdown(
        '<div class="section-title">📚 Your document library</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Browse the PDFs in your index and see how much content "
        "has been indexed from each document."
    )

    if st.button("🔄 Refresh library"):
        st.rerun()

    # Retrieve metadata for all indexed chunks
    with st.spinner("Loading your document library..."):
        library_data = collection.get(include=["metadatas"])

    metadatas = library_data["metadatas"] or []

    # Group chunks by filename
    library = {}

    for metadata in metadatas:
        filename = metadata.get("filename", "Unknown file")
        page_num = metadata.get("page")

        if filename not in library:
            library[filename] = {
                "chunks": 0,
                "pages": set()
            }

        library[filename]["chunks"] += 1

        if page_num is not None:
            library[filename]["pages"].add(page_num)

    if not library:
        st.info(
            "Your library is empty. Upload a PDF to get started."
        )
    else:
        # Search/filter the document list
        filter_text = st.text_input(
            "🔎 Find a document",
            placeholder="Search by filename..."
        )

        filtered_library = {
            name: info
            for name, info in sorted(
                library.items(),
                key=lambda item: item[0].lower()
            )
            if filter_text.lower() in name.lower()
        }

        # Summary metrics
        total_files = len(library)
        total_chunks = sum(
            info["chunks"] for info in library.values()
        )
        total_pages = sum(
            len(info["pages"]) for info in library.values()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("📄 Indexed PDFs", f"{total_files:,}")
        c2.metric("🧩 Text chunks", f"{total_chunks:,}")
        c3.metric("📑 Indexed page references", f"{total_pages:,}")

        st.divider()

        st.subheader(
            f"Documents ({len(filtered_library):,})"
        )

        if not filtered_library:
            st.warning("No documents match your filter.")
        else:
            for filename, info in filtered_library.items():
                with st.container(border=True):
                    left, right = st.columns([4, 1])

                    with left:
                        st.markdown(f"**📄 {filename}**")
                        st.caption(
                            f"{len(info['pages']):,} distinct indexed "
                            f"pages · {info['chunks']:,} text chunks"
                        )

                    with right:
                        if st.button(
                            "Search",
                            key=f"library_search_{filename}",
                            use_container_width=True
                        ):
                            st.session_state["search_query"] = filename
                            st.session_state["library_search_requested"] = True
                            st.rerun()

                    with st.expander("View indexed pages"):
                        pages = sorted(info["pages"])

                        if pages:
                            st.write(
                                ", ".join(str(p) for p in pages)
                            )
                        else:
                            st.caption("No page metadata available.")


# ---------------- UPLOAD PAGE ----------------
if page == "📤 Upload Documents":
    st.markdown('<div class="section-title">Add to your library</div>',
                unsafe_allow_html=True)

    st.write("Upload an academic PDF. New documents are embedded and added to your existing index.")

    uploaded = st.file_uploader(
        "Drop a PDF here or browse your computer",
        type=["pdf"],
        help="Text-based PDFs are supported. Scanned PDFs may need OCR."
    )

    if uploaded is not None:
        st.info(f"Selected: {uploaded.name}")

        if st.button("✨ Upload and index", type="primary",
                    use_container_width=True):
            try:
                with st.spinner("Reading pages and creating embeddings..."):
                    status, count = index_pdf(uploaded)

                if status == "success":
                    st.success(
                        f"Successfully indexed {count:,} passages from {uploaded.name}."
                    )
                    st.rerun()
                elif status == "duplicate":
                    st.info("This exact PDF has already been indexed.")
                else:
                    st.warning(
                        "No selectable text was found. This PDF may be scanned."
                    )

            except Exception as e:
                st.error(f"Indexing failed: {e}")


# ---------------- SEARCH PAGE ----------------
elif page == "🔎 Smart Search":
    st.markdown('<div class="section-title">🔎 Explore your notes</div>',
                unsafe_allow_html=True)

    # Session state preserves the selected query across reruns.
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    
    if st.session_state.pop("library_search_requested", False):
        st.session_state["run_search"] = True

    st.markdown("#### What would you like to learn today?")

    with st.form("search_form"):
        query = st.text_input(
            "Search",
            value=st.session_state.search_query,
            placeholder="e.g. Explain Dijkstra's shortest path algorithm",
            label_visibility="collapsed"
        )

        col1, col2 = st.columns([4, 1])
        with col1:
            top_k = st.slider(
                "Results to display",
                min_value=1,
                max_value=10,
                value=5
            )
        with col2:
            st.markdown("<div style='height: 27px'></div>",
                        unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "✨ Search",
                type="primary",
                use_container_width=True
            )

    st.caption("Try a topic, a full question, or a phrase from your notes.")

    st.markdown("##### 💡 Try an example")

    examples = [
        "Dijkstra algorithm",
        "Database normalization",
        "Round robin scheduling"
    ]

    example_cols = st.columns(3)

    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(example, key=f"example_{i}",
                        use_container_width=True):
                st.session_state.search_query = example
                st.rerun()

    if submitted or st.session_state.pop("run_search", False):
        st.session_state.search_query = query.strip()

        if not query.strip():
            st.warning("Please enter a search question.")
        elif total_chunks == 0:
            st.warning("Your library is empty. Upload a PDF first.")
        else:
            with st.spinner("Finding and ranking relevant passages..."):
                try:
                    query_embedding = model.encode(query).tolist()

                    # Retrieve candidates before reranking.
                    candidate_count = min(50, total_chunks)

                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=candidate_count,
                        include=["documents", "metadatas", "distances"]
                    )

                    docs = results["documents"][0] or []
                    metas = results["metadatas"][0] or []
                    distances = results["distances"][0] or []

                    if not docs:
                        st.warning("No results found. Try another query.")
                    else:
                        tokenized_docs = [tokenize(doc) for doc in docs]
                        bm25 = BM25Okapi(tokenized_docs)
                        keyword_scores = bm25.get_scores(tokenize(query))
                        max_keyword = max(keyword_scores) if len(keyword_scores) else 0

                        ranked = []

                        for i, (doc, meta, distance) in enumerate(
                            zip(docs, metas, distances)
                        ):
                            keyword_score = (
                                float(keyword_scores[i]) / max_keyword
                                if max_keyword > 0 else 0
                            )

                            semantic_score = 1 / (1 + max(float(distance), 0))

                            combined_score = (
                                0.70 * keyword_score
                                + 0.30 * semantic_score
                            )

                            ranked.append((
                                combined_score, doc, meta,
                                float(distance), keyword_score
                            ))

                        ranked.sort(key=lambda item: item[0], reverse=True)

                        st.markdown(
                            f'<div class="section-title">📌 Search results '
                            f'<span style="color:#93c5fd;font-size:0.9rem">'
                            f'({min(top_k, len(ranked))} passages)</span></div>',
                            unsafe_allow_html=True
                        )

                        for i, (score, document, metadata, distance,
                                keyword_score) in enumerate(ranked[:top_k], start=1):
                            filename = metadata.get("filename", "Unknown file")
                            page_num = metadata.get("page", "?")

                            with st.expander(
                                f"📄 {i}. {filename}  ·  Page {page_num}"
                            ):
                                st.markdown(
                                    f"**Source:** `{filename}`  ·  "
                                    f"**Page:** `{page_num}`"
                                )

                                snippet = re.sub(
                                    r"\s+", " ", document
                                ).strip()

                                st.write(snippet[:1000])

                                st.caption(
                                    f"Hybrid score: {score:.3f}  ·  "
                                    f"Keyword score: {keyword_score:.3f}  ·  "
                                    f"Vector distance: {distance:.4f}"
                                )

                except Exception as e:
                    st.error(f"Search failed: {e}")


# ---------------- ABOUT PAGE ----------------
elif page == "ℹ️ About":
    st.markdown('<div class="section-title">About StudyVault AI</div>',
                unsafe_allow_html=True)

    st.write("""
    StudyVault AI is an academic file-search assistant built with
    Python, Streamlit, Sentence Transformers, and ChromaDB.

    **What it does**
    - Searches text extracted from academic PDFs.
    - Finds semantically similar passages.
    - Reranks candidate passages using BM25 keyword scores.
    - Shows filename and page references.
    - Supports incremental indexing through the upload page.

    **Current limitations**
    - Scanned PDFs need OCR.
    - Search quality depends on retrieval and ranking.
    - Page citations identify the source page, not necessarily the exact answer.
    """)


# ---------------- FOOTER ----------------
st.divider()
st.markdown("""
<div class="footer">
    📚 StudyVault AI · Built for smarter learning<br>
    Your notes. Your knowledge. One search away.
</div>
""", unsafe_allow_html=True)
