import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

st.set_page_config(page_title="Cortex AI: RAG Arena", layout="wide", page_icon="🐼  ")
st.title("Cortex AI Testing on RAG")
st.caption("Benchmark Naive Similarity Retrieval vs. Query Expansion + Cross-Encoder Reranking.")

DB_DIR = "./chroma_advanced_db"
LLM_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"

@st.cache_resource
def load_ai_models():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://localhost:11434")
    llm = ChatOllama(model=LLM_MODEL, base_url="http://localhost:11434", temperature=0.1, num_ctx=2048)
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    return embeddings, llm, reranker

embeddings, llm, reranker = load_ai_models()

with st.sidebar:
    st.header("Ingest Course Materials")
    uploaded_file = st.file_uploader("Upload PDF (Syllabus, Handbooks, Rules)", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Index Document", use_container_width=True):
            with st.spinner("Chunking text & computing embeddings..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = text_splitter.split_documents(docs)
                    
                    vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=DB_DIR
                    )
                    st.success(f"Indexed {len(chunks)} chunks into ChromaDB!")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    st.markdown("---")
    st.markdown("### Hardware Monitor")
    st.info("VRAM Target: ~3.5 GB / 6.0 GB (RTX 3060)\nModel: Gemma 3 4B + BGE Reranker")

def run_naive_rag(query: str):
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retrieved_docs = db.similarity_search(query, k=3)
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"Answer strictly using the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    return response.content, retrieved_docs

def run_advanced_rag(query: str):
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    expansion_prompt = (
        f"Generate 2 concise alternative search queries for the question below to help document retrieval.\n"
        f"Return ONLY the 2 queries separated by a newline. Do not include introductory text.\n\n"
        f"Question: {query}"
    )
    raw_expansions = str(llm.invoke(expansion_prompt).content)
    queries = [query] + [q.strip("- ").strip() for q in raw_expansions.split("\n") if q.strip()]
    
    candidates = []
    seen = set()
    for q in queries[:3]:
        for doc in db.similarity_search(q, k=4):
            if doc.page_content not in seen:
                candidates.append(doc)
                seen.add(doc.page_content)
                
    if not candidates:
        return "No documents found.", [], queries

    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_k_docs = [doc for score, doc in scored_docs[:3]]
    
    context = "\n\n".join([doc.page_content for doc in top_k_docs])
    prompt = f"You are Cortex Assistant. Answer the question using this context:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    return response.content, top_k_docs, queries

user_query = st.text_input("💬 Ask a question about your uploaded document:")

if user_query:
    if not os.path.exists(DB_DIR):
        st.warning("Please upload and index a PDF in the sidebar first!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Naive Vector RAG")
            with st.spinner("Running Naive RAG..."):
                n_ans, n_docs = run_naive_rag(user_query)
                st.write(n_ans)
                with st.expander("Retrieved Chunks (Raw Cosine Distance)"):
                    for i, doc in enumerate(n_docs):
                        st.caption(f"Chunk {i+1} | Page {doc.metadata.get('page', 0) + 1}")
                        st.code(doc.page_content, language="text")
                        
        with col2:
            st.subheader("2. Advanced RAG (Expansion + Cross-Encoder)")
            with st.spinner("Rewriting & Reranking..."):
                a_ans, a_docs, expanded_q = run_advanced_rag(user_query)
                st.write(a_ans)
                with st.expander("Pipeline Telemetry & Reranked Matches"):
                    st.write("**Generated Query Variations:**")
                    for eq in expanded_q:
                        st.markdown(f"- `{eq}`")
                    st.write("**Top Reranked Context Chunks:**")
                    for i, doc in enumerate(a_docs):
                        st.caption(f"Rank {i+1} Match | Page {doc.metadata.get('page', 0) + 1}")
                        st.code(doc.page_content, language="text")