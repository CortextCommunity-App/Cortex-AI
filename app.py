import os
import tempfile

import streamlit as st

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Cortex Document Chatbot",
    layout="wide",
)


# ============================================================
# Application configuration
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"
DB_DIR = "./chroma_doc_chat"
COLLECTION_NAME = "cortex_single_rag"

LLM_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"

MAX_HISTORY_MESSAGES = 10
RETRIEVAL_RESULT_COUNT = 4


# ============================================================
# Load and cache application resources
# ============================================================

@st.cache_resource
def load_resources():
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_ctx=4096,
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    return llm, vector_store


try:
    llm, vector_store = load_resources()
except Exception as error:
    st.error(f"Failed to initialize the AI resources: {error}")
    st.stop()


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome to Cortex Document AI. "
                "Upload a PDF from the sidebar, then ask questions "
                "about its content."
            ),
        }
    ]

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ============================================================
# Helper functions
# ============================================================

def reset_conversation():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Conversation reset. What would you like to know "
                "about the uploaded documents?"
            ),
        }
    ]


def format_source(document) -> str:
    filename = document.metadata.get(
        "filename",
        document.metadata.get("source", "Uploaded PDF"),
    )

    page = document.metadata.get("page")

    if isinstance(page, int):
        return f"{filename} (Page {page + 1})"

    return str(filename)


def build_conversation_messages(
    system_instruction: str,
    current_prompt: str,
):
    messages = [
        SystemMessage(content=system_instruction)
    ]

    # Limit history so it does not consume the entire model context.
    recent_history = st.session_state.messages[
        -MAX_HISTORY_MESSAGES:
    ]

    for previous_message in recent_history:
        role = previous_message.get("role")
        content = previous_message.get("content", "")

        if not content:
            continue

        if role == "user":
            messages.append(
                HumanMessage(content=content)
            )
        elif role == "assistant":
            messages.append(
                AIMessage(content=content)
            )

    messages.append(
        HumanMessage(content=current_prompt)
    )

    return messages


def display_sources(sources):
    if not sources:
        return

    with st.expander("View retrieved sources"):
        for source in sources:
            st.markdown(f"**{source['source']}**")
            st.write(source["content"])
            st.divider()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("Document Hub")

    st.caption(
        "Upload course syllabi, handbooks, or research papers."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
    )

    if uploaded_file is not None:
        if st.button(
            "Index document",
            use_container_width=True,
        ):
            temp_path = None

            try:
                file_data = uploaded_file.getvalue()

                if not file_data:
                    st.error("The uploaded PDF is empty.")
                elif not file_data.startswith(b"%PDF"):
                    st.error(
                        "The uploaded file does not contain valid PDF data."
                    )
                else:
                    with st.spinner(
                        "Reading and indexing the document..."
                    ):
                        with tempfile.NamedTemporaryFile(
                            mode="wb",
                            delete=False,
                            suffix=".pdf",
                        ) as temporary_file:
                            temporary_file.write(file_data)
                            temp_path = temporary_file.name

                        loader = PyPDFLoader(temp_path)
                        pages = loader.load()

                        if not pages:
                            raise ValueError(
                                "No readable pages were found."
                            )

                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=50,
                            length_function=len,
                            separators=[
                                "\n\n",
                                "\n",
                                ". ",
                                " ",
                                "",
                            ],
                        )

                        chunks = splitter.split_documents(pages)
                        valid_chunks = []

                        for chunk_index, chunk in enumerate(chunks):
                            content = chunk.page_content.strip()

                            if not content:
                                continue

                            chunk.page_content = content
                            chunk.metadata["filename"] = (
                                uploaded_file.name
                            )
                            chunk.metadata["chunk_index"] = (
                                chunk_index
                            )

                            valid_chunks.append(chunk)

                        if not valid_chunks:
                            raise ValueError(
                                "The PDF contains no extractable text."
                            )

                        vector_store.add_documents(valid_chunks)

                        if (
                            uploaded_file.name
                            not in st.session_state.indexed_files
                        ):
                            st.session_state.indexed_files.append(
                                uploaded_file.name
                            )

                    st.success(
                        f"Indexed {len(valid_chunks)} chunks from "
                        f"{uploaded_file.name}."
                    )

            except Exception as error:
                st.error(
                    f"Document indexing failed: {error}"
                )

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        reset_conversation()
        st.rerun()

    st.divider()

    st.subheader("Indexed in this session")

    if st.session_state.indexed_files:
        for filename in st.session_state.indexed_files:
            st.write(filename)
    else:
        st.caption("No documents indexed in this session.")

    st.divider()

    st.subheader("Local engine")

    st.markdown(
        f"""
**Model:** {LLM_MODEL}

**Embeddings:** {EMBEDDING_MODEL}

**Storage:** ChromaDB Local
"""
    )


# ============================================================
# Main interface
# ============================================================

st.title("Cortex PDF Copilot")

st.caption(
    "Ask questions about uploaded documents with conversational "
    "history and source references."
)


# ============================================================
# Display conversation
# ============================================================

for message in st.session_state.messages:
    role = message.get("role", "assistant")
    content = message.get("content", "")

    with st.chat_message(role):
        st.markdown(content)

        display_sources(
            message.get("sources", [])
        )


# ============================================================
# Chat processing
# ============================================================

prompt = st.chat_input(
    "Ask a question about the uploaded documents..."
)

if prompt:
    normalized_prompt = prompt.strip()

    if not normalized_prompt:
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": normalized_prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(normalized_prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents..."):
                retrieved_documents = (
                    vector_store.similarity_search(
                        normalized_prompt,
                        k=RETRIEVAL_RESULT_COUNT,
                    )
                )

            if not retrieved_documents:
                bot_answer = (
                    "I could not find relevant information in the "
                    "indexed documents. Upload and index a PDF before "
                    "asking document-specific questions."
                )
                sources = []

            else:
                context_sections = []
                sources = []

                for index, document in enumerate(
                    retrieved_documents,
                    start=1,
                ):
                    source_name = format_source(document)

                    context_sections.append(
                        f"[Passage {index} | {source_name}]\n"
                        f"{document.page_content}"
                    )

                    sources.append(
                        {
                            "source": source_name,
                            "content": document.page_content,
                        }
                    )

                context_text = "\n\n".join(context_sections)

                system_instruction = f"""
You are Cortex Assistant, an academic document tutor.

Answer the user's question using only the supplied document context.

Rules:
- Do not use unsupported outside knowledge.
- If the answer is absent, say that it is not present in the documents.
- Do not invent or extrapolate information.
- Cite sources using the document name and page number.
- Use previous conversation messages only to understand references
  such as "it", "that", or "the previous topic".
- The retrieved document context is the factual source of truth.

DOCUMENT CONTEXT:

{context_text}
""".strip()

                conversation_messages = (
                    build_conversation_messages(
                        system_instruction=system_instruction,
                        current_prompt=normalized_prompt,
                    )
                )

                with st.spinner("Generating an answer..."):
                    response = llm.invoke(
                        conversation_messages
                    )

                bot_answer = str(response.content)

            st.markdown(bot_answer)
            display_sources(sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": bot_answer,
                    "sources": sources,
                }
            )

        except Exception as error:
            error_message = (
                f"Unable to generate an answer: {error}"
            )

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )