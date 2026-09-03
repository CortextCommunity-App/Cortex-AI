import base64
import os
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI(
    title="Cortex AI Engine",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma3:4b"

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)

llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    num_ctx=4096,
)

CHROMA_DIR = "./chroma_data"
COLLECTION_NAME = "cortex_knowledge"

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

indexed_docs: list[Document] = []

conversation_store: dict[str, deque[BaseMessage]] = defaultdict(
    lambda: deque(maxlen=20)
)

conversation_lock = RLock()
class SimpleAskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=4000,
        description="A standalone question sent directly to the model.",
    )

def get_conversation_history(session_id: str) -> list[BaseMessage]:
    """Return a safe copy of a session's conversation history."""
    with conversation_lock:
        return list(conversation_store[session_id])


def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Save one user-assistant conversation turn."""
    with conversation_lock:
        conversation_store[session_id].append(
            HumanMessage(content=user_message)
        )
        conversation_store[session_id].append(
            AIMessage(content=assistant_message)
        )


def load_documents_from_chroma() -> list[Document]:
    """
    Reload persisted Chroma documents so BM25 retrieval still
    works after restarting FastAPI.
    """
    try:
        stored_data = vector_store.get(
            include=["documents", "metadatas"]
        )

        stored_documents = stored_data.get("documents") or []
        stored_metadatas = stored_data.get("metadatas") or []

        loaded_documents: list[Document] = []

        for index, content in enumerate(stored_documents):
            if not content:
                continue

            metadata = {}

            if index < len(stored_metadatas):
                metadata = stored_metadatas[index] or {}

            loaded_documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        return loaded_documents

    except Exception as error:
        print(f"Could not reload documents from Chroma: {error}")
        return []


indexed_docs.extend(load_documents_from_chroma())


def format_retrieved_documents(
    documents: list[Document],
) -> str:
    """Format retrieved documents with source information."""
    formatted_results: list[str] = []

    for index, document in enumerate(documents, start=1):
        filename = document.metadata.get(
            "filename",
            document.metadata.get("source", "Unknown source"),
        )

        page_number = document.metadata.get("page")

        if isinstance(page_number, int):
            source = f"{filename}, page {page_number + 1}"
        else:
            source = str(filename)

        formatted_results.append(
            f"[Result {index} | Source: {source}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted_results)

@tool
def search_community_knowledge(query: str) -> str:
    """
    Search uploaded PDFs, course syllabi, and technical handbooks
    using semantic and keyword retrieval.
    """
    normalized_query = query.strip()

    if not normalized_query:
        return "A search query is required."

    # Fall back to semantic search if BM25 has no documents.
    if not indexed_docs:
        results = vector_store.similarity_search(
            normalized_query,
            k=4,
        )

        if not results:
            return "No documents were found in the knowledge base."

        return format_retrieved_documents(results)

    chroma_retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    bm25_retriever = BM25Retriever.from_documents(indexed_docs)
    bm25_retriever.k = 4

    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            chroma_retriever,
        ],
        weights=[0.4, 0.6],
    )

    retrieved_documents = ensemble_retriever.invoke(
        normalized_query
    )

    if not retrieved_documents:
        return "No matching content was found."

    return format_retrieved_documents(retrieved_documents)


@tool
def query_event_status(event_name: str) -> str:
    """
    Check the status, venue, and check-in requirements for
    community hackathons or workshops.
    """
    events_mock_database = {
        "code-c": (
            "CODE-C Hackathon is ACTIVE. "
            "Venue: SETEC Hall B. "
            "Check-in requires a valid QR pass."
        ),
        "workshop": (
            "Spring Boot & AI Workshop starts Saturday at "
            "9:00 AM in Room 302."
        ),
    }

    normalized_name = event_name.lower().strip()

    if not normalized_name:
        return "An event name is required."

    for event_key, event_details in events_mock_database.items():
        if event_key in normalized_name:
            return event_details

    return f"No active record was found for event: '{event_name}'."


tools = [
    search_community_knowledge,
    query_event_status,
]

tool_map = {
    current_tool.name: current_tool
    for current_tool in tools
}

llm_with_tools = llm.bind_tools(tools)

class AgentQueryRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique conversation session identifier.",
    )

    query: str = Field(
        min_length=1,
        max_length=4000,
        description="Question submitted to the Cortex agent.",
    )


@app.post("/api/v1/ai/upload-doc")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Parse a PDF, split it into chunks, and index it using
    Chroma and BM25.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must have a filename.",
        )

    safe_filename = Path(file.filename).name

    if Path(safe_filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    allowed_content_types = {
        "application/pdf",
        "application/octet-stream",
    }

    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not recognized as a PDF.",
        )

    temp_path: str | None = None

    try:
        file_contents = await file.read()

        if not file_contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty.",
            )

        if not file_contents.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file does not contain valid PDF data.",
            )

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".pdf",
            delete=False,
        ) as temporary_file:
            temporary_file.write(file_contents)
            temp_path = temporary_file.name

        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        if not pages:
            raise HTTPException(
                status_code=422,
                detail="No readable pages were found in the PDF.",
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=40,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(pages)
        valid_chunks: list[Document] = []

        for chunk_index, chunk in enumerate(chunks):
            content = chunk.page_content.strip()

            if not content:
                continue

            chunk.page_content = content
            chunk.metadata["filename"] = safe_filename
            chunk.metadata["chunk_index"] = chunk_index

            valid_chunks.append(chunk)

        if not valid_chunks:
            raise HTTPException(
                status_code=422,
                detail="The PDF did not contain extractable text.",
            )

        vector_store.add_documents(valid_chunks)
        indexed_docs.extend(valid_chunks)

        return {
            "status": "success",
            "filename": safe_filename,
            "pages_read": len(pages),
            "chunks_indexed": len(valid_chunks),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {error}",
        ) from error

    finally:
        await file.close()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


AGENT_SYSTEM_PROMPT = """
You are Cortex Agent, the official AI assistant for Cortex Community at SETEC Institute.

### CAPABILITIES & TOOLS:
1. `search_community_knowledge`: Query course syllabi, lecture slides, and institutional handbooks.
2. `query_event_status`: Check hackathons, workshops, and check-in requirements.

### OPERATIONAL RULES:
- GENERAL KNOWLEDGE: For general programming, computer science concepts (Java, OOP, Python), or casual chat, answer directly using your own knowledge without calling tools.
- MULTILINGUAL: Respond in the language requested by the user (including Khmer ភាសាខ្មែរ).
- INSTITUTIONAL GROUNDING: For questions specific to SETEC, Cortex Community, or uploaded course files, rely strictly on tools and cite sources.
- CONVERSATION CONTEXT: Evaluate the chat history to resolve references and follow-ups.
- NEVER HALLUCINATE: If a tool returns no data for a specific document or event, state that clearly.
""".strip()

@app.post("/api/v1/ai/message")
def simple_ask(request: SimpleAskRequest):
    """
    Send a standalone question directly to the model.

    This endpoint:
    - Does not use conversation history.
    - Does not save messages.
    - Does not use agent tools.
    - Does not search uploaded documents.
    """
    try:
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful AI assistant. "
                    "Answer the user's question clearly and accurately. "
                    "If you do not know the answer, say that you do not know."
                )
            ),
            HumanMessage(content=request.query.strip()),
        ]

        response = llm.invoke(messages)

        return {
            "answer": response.content,
            "model": CHAT_MODEL,
            "conversation_saved": False,
            "tools_invoked": [],
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Simple question failed: {error}",
        ) from error

@app.post("/api/v1/ai/agent-ask")
def agent_ask(request: AgentQueryRequest):
    """
    Process a conversation using chat history and an agent-tool loop.
    """
    user_query = request.query.strip()

    conversation_history = get_conversation_history(
        request.session_id
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        *conversation_history,
        HumanMessage(content=user_query),
    ]

    invoked_tools: list[str] = []
    maximum_tool_steps = 3

    try:
        for _ in range(maximum_tool_steps):
            ai_message = llm_with_tools.invoke(messages)
            messages.append(ai_message)

            tool_calls = getattr(
                ai_message,
                "tool_calls",
                [],
            )

            if not tool_calls:
                answer = str(ai_message.content)

                save_conversation_turn(
                    session_id=request.session_id,
                    user_message=user_query,
                    assistant_message=answer,
                )

                return {
                    "answer": answer,
                    "session_id": request.session_id,
                    "agent_invoked_tools": invoked_tools,
                }

            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_arguments: dict[str, Any] = tool_call.get(
                    "args",
                    {},
                )
                tool_call_id = tool_call["id"]

                invoked_tools.append(tool_name)

                selected_tool = tool_map.get(tool_name)

                if selected_tool is None:
                    tool_result = (
                        f"Unknown tool requested: {tool_name}"
                    )
                else:
                    try:
                        tool_result = selected_tool.invoke(
                            tool_arguments
                        )
                    except Exception as tool_error:
                        tool_result = (
                            f"Tool '{tool_name}' failed: "
                            f"{tool_error}"
                        )

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id,
                    )
                )

        messages.append(
            HumanMessage(
                content=(
                    "You have reached the tool-call limit. "
                    "Do not call another tool. Provide a final answer "
                    "using the information already available."
                )
            )
        )

        final_response = llm.invoke(messages)
        final_answer = str(final_response.content)

        save_conversation_turn(
            session_id=request.session_id,
            user_message=user_query,
            assistant_message=final_answer,
        )

        return {
            "answer": final_answer,
            "session_id": request.session_id,
            "agent_invoked_tools": invoked_tools,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {error}",
        ) from error



@app.get("/api/v1/ai/conversations/{session_id}")
def get_conversation(session_id: str):
    """Return the stored history for one session."""
    with conversation_lock:
        history = list(conversation_store.get(session_id, []))

    formatted_history = []

    for message in history:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "unknown"

        formatted_history.append(
            {
                "role": role,
                "content": message.content,
            }
        )

    return {
        "session_id": session_id,
        "message_count": len(formatted_history),
        "messages": formatted_history,
    }


@app.delete("/api/v1/ai/conversations/{session_id}")
def clear_conversation(session_id: str):
    """Delete the conversation history for one session."""
    with conversation_lock:
        existed = session_id in conversation_store
        conversation_store.pop(session_id, None)

    return {
        "status": "cleared",
        "session_id": session_id,
        "existed": existed,
    }

@app.post("/api/v1/ai/analyze-image")
async def analyze_image(
    prompt: str = Form(
        default="Analyze this image or error screenshot."
    ),
    image: UploadFile = File(...),
):
    """Analyze an uploaded image using Gemma 3."""
    allowed_content_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if image.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG, and WebP images are supported.",
        )

    try:
        image_contents = await image.read()

        if not image_contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty.",
            )

        encoded_image = base64.b64encode(
            image_contents
        ).decode("utf-8")

        image_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        prompt.strip()
                        or "Analyze this image."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": (
                        f"data:{image.content_type};"
                        f"base64,{encoded_image}"
                    ),
                },
            ]
        )

        response = llm.invoke([image_message])

        return {
            "analysis": response.content,
            "model": CHAT_MODEL,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {error}",
        ) from error

    finally:
        await image.close()

@app.get("/health")
def health():
    with conversation_lock:
        active_conversations = len(conversation_store)

    return {
        "status": "ok",
        "architecture": "Agentic Hybrid RAG",
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "indexed_documents": len(indexed_docs),
        "active_conversations": active_conversations,
    }