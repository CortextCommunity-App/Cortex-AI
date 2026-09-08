import base64
import os
import re
import socket
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Force IPv4 resolution for host/localhost connections to prevent [Errno 101] Network is unreachable on Docker
_orig_getaddrinfo = socket.getaddrinfo

def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host in ("host.docker.internal", "localhost", "127.0.0.1"):
        return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
    return _orig_getaddrinfo(host, port, family, type, proto, flags)

socket.getaddrinfo = _ipv4_getaddrinfo

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
    version="1.0.0",
    description="Academic AI Microservice for Cortex Community at SETEC Institute",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"AI Engine internal error: {str(exc)}"},
        headers={"Access-Control-Allow-Origin": "*"},
    )

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma3:4b")
VISION_MODEL = os.getenv("VISION_MODEL", "gemma3:4b")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)

llm_strict = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    num_ctx=4096,
)

llm_creative = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7,
    num_ctx=4096,
)

llm = llm_strict

vision_llm = ChatOllama(
    model=VISION_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cortex_knowledge")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

indexed_docs: list[Document] = []
conversation_store: dict[str, deque[BaseMessage]] = defaultdict(lambda: deque(maxlen=20))
conversation_lock = RLock()


def get_conversation_history(session_id: str) -> list[BaseMessage]:
    with conversation_lock:
        return list(conversation_store[session_id])


def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    with conversation_lock:
        conversation_store[session_id].append(HumanMessage(content=user_message))
        conversation_store[session_id].append(AIMessage(content=assistant_message))


def load_documents_from_chroma() -> list[Document]:
    try:
        stored_data = vector_store.get(include=["documents", "metadatas"])
        stored_documents = stored_data.get("documents") or []
        stored_metadatas = stored_data.get("metadatas") or []

        loaded_documents: list[Document] = []
        for index, content in enumerate(stored_documents):
            if not content:
                continue
            metadata = stored_metadatas[index] if index < len(stored_metadatas) else {}
            loaded_documents.append(
                Document(page_content=content, metadata=metadata or {})
            )
        return loaded_documents
    except Exception as error:
        print(f"Chroma document reload warning: {error}")
        return []


indexed_docs.extend(load_documents_from_chroma())


def format_retrieved_documents(documents: list[Document]) -> str:
    formatted_results: list[str] = []
    for index, document in enumerate(documents, start=1):
        filename = document.metadata.get(
            "filename",
            document.metadata.get("source", "Course Document"),
        )
        page_number = document.metadata.get("page")
        source = f"{filename}, page {page_number + 1}" if isinstance(page_number, int) else str(filename)
        formatted_results.append(f"[Result {index} | Source: {source}]\n{document.page_content}")
    return "\n\n".join(formatted_results)


@tool
def search_community_knowledge(query: str) -> str:
    """Search uploaded course syllabi, handbooks, and PDFs using hybrid retrieval."""
    normalized_query = query.strip()
    if not normalized_query:
        return "A search query is required."

    if not indexed_docs:
        results = vector_store.similarity_search(normalized_query, k=4)
        if not results:
            return "No documents found in the knowledge base."
        return format_retrieved_documents(results)

    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = BM25Retriever.from_documents(indexed_docs)
    bm25_retriever.k = 4

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6],
    )
    retrieved_documents = ensemble_retriever.invoke(normalized_query)
    if not retrieved_documents:
        return "No matching content found."
    return format_retrieved_documents(retrieved_documents)


@tool
def query_event_status(event_name: str) -> str:
    """Check status, venue, and check-in details for hackathons and workshops."""
    events_mock_database = {
        "code-c": (
            "CODE-C Hackathon is ACTIVE. Venue: SETEC Hall B. "
            "Check-in requires a valid QR pass."
        ),
        "workshop": (
            "Spring Boot & AI Workshop starts Saturday at 9:00 AM in Room 302."
        ),
    }
    normalized_name = event_name.lower().strip()
    if not normalized_name:
        return "An event name is required."

    for event_key, event_details in events_mock_database.items():
        if event_key in normalized_name:
            return event_details
    return f"No active record found for event: '{event_name}'."


tools = [search_community_knowledge, query_event_status]
tool_map = {t.name: t for t in tools}

try:
    llm_strict_with_tools = llm_strict.bind_tools(tools)
except Exception:
    llm_strict_with_tools = llm_strict

try:
    llm_creative_with_tools = llm_creative.bind_tools(tools)
except Exception:
    llm_creative_with_tools = llm_creative

llm_with_tools = llm_strict_with_tools


def get_llm(creative: bool = False) -> ChatOllama:
    """Returns an adaptive ChatOllama instance: 0.7 for creative, 0.1 for strict."""
    return llm_creative if creative else llm_strict


def get_llm_with_tools(creative: bool = False):
    """Returns an adaptive tool-bound ChatOllama instance."""
    return llm_creative_with_tools if creative else llm_strict_with_tools


CREATIVE_KEYWORDS_PATTERN = re.compile(
    r"\b(brainstorm\w*|story|stories|tell\s+me\s+a\s+story|joke|jokes|tell\s+me\s+a\s+joke|"
    r"analogy|analogies|creative|create\s+a\s+name|project\s+ideas?|idea\s+for|inspire|"
    r"poem|poetry|riddle|fun\s+fact|imagine|fiction|casual|chat|conversation|hobby|hobbies|"
    r"opinion|how\s+are\s+you|favorite\s+movie|favorite\s+music)\b",
    re.IGNORECASE,
)

FACTUAL_STRICT_PATTERN = re.compile(
    r"\b(founder|team|creator|who\s+made|who\s+built|syllabus|handbook|pdf|document|"
    r"code-c|hackathon|venue|schedule|time|date|verify|exact|true\s+or\s+false|"
    r"fix\s+bug|syntax\s+error|debug|stack\s+trace|exception|regex|sql\s+query)\b",
    re.IGNORECASE,
)


def is_creative_query(query: str) -> bool:
    """
    Determines whether a query is open-ended/creative (temp=0.7)
    or strictly factual/deterministic (temp=0.1).
    """
    if FACTUAL_STRICT_PATTERN.search(query):
        return False
    return bool(CREATIVE_KEYWORDS_PATTERN.search(query))

SAFETY_REFUSAL_MESSAGE = "I cannot fulfill this request as it violates academic community guidelines."

SAFETY_INTERVENTION_NOTICE = "The generated response was blocked by Cortex safety policies."

PROHIBITED_SAFETY_PATTERNS = [
    r"\b(porn|porno|pornography|xxx|nsfw|nude\w*|erotic|erotica|hentai|sex\s+story|porn\s+story|sexually\s+explicit)\b",
    r"\b(blowjob|handjob|cumshot|gangbang|deepthroat|masturbat\w*|orgasm\w*|penis|vagina|vulva|genital\w*)\b",
    r"\b(incest|rape|raping|pedophil\w*|child\s+porn\w*|pedosex\w*)\b",
    r"\b(naked\s+women|naked\s+men|naked\s+people)\b",

    r"\b(how\s+to\s+(commit\s+suicide|kill\s+myself|hang\s+myself))\b",
    r"\b(make|build|manufacture)\s+(a\s+)?(bomb|explosive|dirty\s+bomb)\b",
]

COMPILED_SAFETY_REGEXES = [re.compile(p, re.IGNORECASE) for p in PROHIBITED_SAFETY_PATTERNS]


def check_content_safety(text: str) -> bool:
    """
    Validates input and output content against safety policies.
    Returns True if content is safe.
    Returns False if it matches prohibited adult/NSFW or severe harm concepts.
    """
    if not text:
        return True
    for pattern in COMPILED_SAFETY_REGEXES:
        if pattern.search(text):
            return False
    return True


def sanitize_model_output(text: str) -> str:
    """Replaces unsafe model output with a standard safety intervention notice."""
    if not check_content_safety(text):
        return SAFETY_INTERVENTION_NOTICE
    return text


class SimpleAskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    creative: bool | None = None


class AgentQueryRequest(BaseModel):
    session_id: str = Field(default="default_session", min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, Any]] | None = None
    creative: bool | None = None


GREETING_PATTERNS = {
    "hi", "hello", "hey", "who are you", "who are you?",
    "hi who are you", "hi who are you?", "good morning", "good evening"
}

AGENT_SYSTEM_PROMPT = """
You are Cortex Copilot, the official AI academic and community assistant for Cortex Community at SETEC Institute.
Your role is to be helpful, polite, and conversational.

CAPABILITIES & SCOPE:
1. Academic & Technical: Assist students with programming concepts, debugging, course curriculum, homework explanations, and software development.
2. Community & Everyday Chat: Engage warmly in friendly everyday conversation, personal hobbies, career guidance, and general knowledge questions without unnecessary restrictions.
3. Language: Support English and Khmer (ភាសាខ្មែរ) fluently.

FOUNDERS & CORE TEAM:
- Founder & Lead Software Engineer: Pheng Mengheak (Project vision, full-stack architecture, and AI microservices).
- UI/UX Designers: Sokha Neasa (Lead UI/UX Designer), Neao Sivlin (UI/UX Designer).
- Frontend Engineers: Thy Maya (Frontend Architecture & Client Implementation), Vin Sreynich (វិន ស្រីនិច) (Frontend Engineer).

TEAM & PERSON QUERY RULES:
1. Individual Focus: If the user asks about a specific person (e.g., Thy Maya, Sokha Neasa, Neao Sivlin, Vin Sreynich, or Pheng Mengheak), focus your response exclusively on that person and their role, rather than dumping the whole team list unless explicitly asked for the full roster.
2. Full Roster: If the user asks generally about the founders, creators, or the team who built Cortex Community, provide credit to Pheng Mengheak and the core team members directly without calling tools.

SAFETY & BEHAVIORAL BOUNDARIES:
1. Hard Boundaries: Strictly decline requests involving sexually explicit content, pornography, severe violence, or direct instructions for physical harm or weapons. If violated, politely state community boundaries.
2. Clean Output: NEVER output pseudo-code or blocks like `tool_code` or `search_community_knowledge(...)` in your final text. Always respond in clear, helpful natural language.
3. Tools: Call search_community_knowledge ONLY when the user asks questions about uploaded documents, handbooks, or course syllabi. Call query_event_status ONLY for campus hackathons or workshops (e.g., CODE-C).
""".strip()

# ============================================================
# API Endpoints (v1)
# ============================================================
@app.post("/api/v1/ai/upload-doc")
async def upload_pdf(file: UploadFile = File(...)):
    """Parse a PDF, chunk it, and index it into ChromaDB and BM25."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="The uploaded file must have a filename.")

    safe_filename = Path(file.filename).name
    if Path(safe_filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path: str | None = None
    try:
        file_contents = await file.read()
        if not file_contents or not file_contents.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="Invalid PDF file contents.")

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as tmp:
            tmp.write(file_contents)
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        if not pages:
            raise HTTPException(status_code=422, detail="No readable text found in PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=40,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(pages)
        valid_chunks: list[Document] = []

        for index, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            if not content:
                continue
            chunk.page_content = content
            chunk.metadata["filename"] = safe_filename
            chunk.metadata["chunk_index"] = index
            valid_chunks.append(chunk)

        if not valid_chunks:
            raise HTTPException(status_code=422, detail="The PDF did not contain extractable text.")

        vector_store.add_documents(valid_chunks)
        indexed_docs.extend(valid_chunks)

        return {
            "status": "success",
            "filename": safe_filename,
            "pages_read": len(pages),
            "chunks_indexed": len(valid_chunks),
        }
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/v1/ai/message")
def simple_ask(request: SimpleAskRequest):
    """Direct chat completion endpoint with dynamic temperature."""
    user_query = request.query.strip()
    if not check_content_safety(user_query):
        return {
            "answer": SAFETY_REFUSAL_MESSAGE,
            "model": CHAT_MODEL,
            "creative": False,
            "tools_invoked": [],
        }

    is_creative = request.creative if request.creative is not None else is_creative_query(user_query)
    current_llm = get_llm(creative=is_creative)

    try:
        messages = [
            SystemMessage(
                content=(
                    "You are Cortex Assistant for SETEC Institute. Answer clearly and concisely. "
                    "Strictly refuse inappropriate, sexually explicit, or unsafe requests."
                )
            ),
            HumanMessage(content=user_query),
        ]
        response = current_llm.invoke(messages)
        output_text = sanitize_model_output(str(response.content))

        return {
            "answer": output_text,
            "model": CHAT_MODEL,
            "creative": is_creative,
            "tools_invoked": [],
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference error: {error}") from error


@app.post("/api/v1/ai/agent-ask")
def agent_ask(request: AgentQueryRequest):
    """Conversational endpoint with guardrails, dynamic temperature, and tool execution."""
    user_query = request.query.strip()
    query_lower = user_query.lower()

    if not check_content_safety(user_query):
        return {
            "answer": SAFETY_REFUSAL_MESSAGE,
            "session_id": request.session_id,
            "creative": False,
            "agent_invoked_tools": [],
        }

    if query_lower in GREETING_PATTERNS:
        greeting = (
            "Hello! I am Cortex Copilot, your academic assistant for SETEC Institute. "
            "I can help answer questions from course syllabi, verify CODE-C event schedules, "
            "or explain programming concepts. How can I assist you today?"
        )
        save_conversation_turn(request.session_id, user_query, greeting)
        return {
            "answer": greeting,
            "session_id": request.session_id,
            "creative": False,
            "agent_invoked_tools": [],
        }

    is_creative = request.creative if request.creative is not None else is_creative_query(user_query)
    active_llm = get_llm(creative=is_creative)
    active_llm_with_tools = get_llm_with_tools(creative=is_creative)

    # 3. Context assembly
    conversation_history = get_conversation_history(request.session_id)
    if not conversation_history and request.history:
        for turn in request.history:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                conversation_history.append(HumanMessage(content=content))
            elif role == "assistant":
                conversation_history.append(AIMessage(content=content))

    messages: list[BaseMessage] = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        *conversation_history,
        HumanMessage(content=user_query),
    ]

    invoked_tools: list[str] = []
    maximum_tool_steps = 3

    try:
        for _ in range(maximum_tool_steps):
            try:
                ai_message = active_llm_with_tools.invoke(messages)
            except Exception:
                ai_message = active_llm.invoke(messages)
                answer = sanitize_model_output(str(ai_message.content))
                save_conversation_turn(request.session_id, user_query, answer)
                return {
                    "answer": answer,
                    "session_id": request.session_id,
                    "creative": is_creative,
                    "agent_invoked_tools": invoked_tools,
                }

            messages.append(ai_message)
            tool_calls = getattr(ai_message, "tool_calls", [])

            if not tool_calls:
                raw_text = str(ai_message.content)
                if "search_community_knowledge" in raw_text and ("tool_code" in raw_text or "query=" in raw_text or "(" in raw_text):
                    match = re.search(r'query=["\']([^"\']+)["\']', raw_text)
                    subquery = match.group(1) if match else user_query
                    invoked_tools.append("search_community_knowledge")
                    try:
                        tool_res = search_community_knowledge.invoke({"query": subquery})
                    except Exception as err:
                        tool_res = f"Retrieval error: {err}"

                    followup_messages = [
                        *messages,
                        ai_message,
                        HumanMessage(
                            content=(
                                f"Knowledge retrieval results:\n{tool_res}\n\n"
                                "Now synthesize a comprehensive, direct answer for the user in natural language. "
                                "Never print tool codes or function call blocks."
                            )
                        ),
                    ]
                    final_resp = llm_strict.invoke(followup_messages)
                    answer = sanitize_model_output(str(final_resp.content))
                else:
                    answer = sanitize_model_output(raw_text)

                save_conversation_turn(request.session_id, user_query, answer)
                return {
                    "answer": answer,
                    "session_id": request.session_id,
                    "creative": is_creative,
                    "agent_invoked_tools": invoked_tools,
                }

            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                tool_id = tool_call["id"]
                invoked_tools.append(tool_name)

                target_tool = tool_map.get(tool_name)
                if target_tool is None:
                    tool_result = f"Unknown tool: {tool_name}"
                else:
                    try:
                        tool_result = target_tool.invoke(tool_args)
                    except Exception as err:
                        tool_result = f"Tool failure: {err}"

                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))

        messages.append(HumanMessage(content="Provide a final answer using the available information."))
        final_response = llm_strict.invoke(messages)
        final_answer = sanitize_model_output(str(final_response.content))
        save_conversation_turn(request.session_id, user_query, final_answer)

        return {
            "answer": final_answer,
            "session_id": request.session_id,
            "creative": is_creative,
            "agent_invoked_tools": invoked_tools,
        }

    except Exception as error:
        try:
            fallback_resp = active_llm.invoke(messages)
            fallback_ans = sanitize_model_output(str(fallback_resp.content))
        except Exception as fallback_error:
            fallback_ans = (
                f"I encountered a temporary communication issue with the local AI model: {fallback_error}. "
                "Please make sure the local Ollama service is running and accessible."
            )
        save_conversation_turn(request.session_id, user_query, fallback_ans)
        return {
            "answer": fallback_ans,
            "session_id": request.session_id,
            "creative": is_creative,
            "agent_invoked_tools": invoked_tools,
        }


@app.get("/api/v1/ai/conversations/{session_id}")
def get_conversation(session_id: str):
    """Retrieve in-memory history for a session."""
    history = get_conversation_history(session_id)
    formatted = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in history
    ]
    return {
        "session_id": session_id,
        "message_count": len(formatted),
        "messages": formatted,
    }


@app.delete("/api/v1/ai/conversations/{session_id}")
def clear_conversation(session_id: str):
    """Clear conversation history for a session."""
    with conversation_lock:
        existed = session_id in conversation_store
        conversation_store.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id, "existed": existed}


@app.post("/api/v1/ai/analyze-image")
async def analyze_image(
    prompt: str = Form(default="Analyze this image or error screenshot."),
    image: UploadFile = File(...),
):
    """Multimodal vision diagnostics via Gemma 3."""
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, and WebP images are supported.")

    user_prompt = prompt.strip() or "What is in this image?"
    if not check_content_safety(user_prompt):
        return {"analysis": SAFETY_REFUSAL_MESSAGE, "model": VISION_MODEL}

    try:
        image_contents = await image.read()
        if not image_contents:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")

        encoded_image = base64.b64encode(image_contents).decode("utf-8")

        vision_instruction = (
            "You are Cortex Copilot Vision, an intelligent multimodal diagnostic assistant for Cortex Community at SETEC Institute. "
            "Analyze the attached image in detail and format your response in clear, well-structured Markdown.\n\n"
            "Key Diagnostic Guidelines:\n"
            "1. Visual Identification: Describe the key objects, user interface elements, architecture diagrams, layouts, or graphics visible.\n"
            "2. Text & Code Recognition: Transcribe and interpret any visible code snippets, error traces, terminal output, or textual labels.\n"
            "3. Answer the User Request: Directly and thoroughly answer whatever question the user asked regarding this picture.\n"
            "4. Practical Guidance: Provide actionable advice or troubleshooting steps if this is a bug, code issue, or technical diagram."
        )

        full_prompt = f"{vision_instruction}\n\nUser Question/Instruction: {user_prompt}"

        image_message = HumanMessage(
            content=[
                {"type": "text", "text": full_prompt},
                {"type": "image_url", "image_url": f"data:{image.content_type};base64,{encoded_image}"},
            ]
        )

        response = vision_llm.invoke([image_message])
        analysis_text = sanitize_model_output(str(response.content))
        return {
            "analysis": analysis_text,
            "model": VISION_MODEL,
        }
    except Exception as err:
        return {
            "analysis": f"Diagnostic analysis error: {err}",
            "model": VISION_MODEL,
        }
    finally:
        await image.close()


@app.get("/health")
def health():
    """Service health and telemetry."""
    with conversation_lock:
        active_conversations = len(conversation_store)
    return {
        "status": "ok",
        "api_version": "1.0.0",
        "architecture": "Agentic Hybrid RAG",
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "indexed_documents": len(indexed_docs),
        "active_conversations": active_conversations,
    }