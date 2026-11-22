import os
import re
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# Load .env from the SAME folder as this file, regardless of where the
# script is launched from (fixes cases where load_dotenv() silently found
# nothing because the current working directory didn't match).
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError(
        f"GOOGLE_API_KEY not found.\n"
        f"Looked for a .env file at: {_ENV_PATH}\n"
        f"Checklist:\n"
        f"  1. Does that exact file exist? (run `dir -Force` in that folder)\n"
        f"  2. Is it named exactly '.env' (NOT '.env.txt' — Windows often "
        f"hides the real extension)?\n"
        f"  3. Does it contain a line like: GOOGLE_API_KEY=AQ.Ab8...\n"
        f"     (no quotes, no spaces around the '=')\n"
        f"  4. Is python-dotenv installed in this venv? "
        f"(`pip show python-dotenv`)"
    )

import fitz
import pytesseract

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VALID_BOOK_TYPES = {"coding", "novel", "manga"}
TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens=2048,  # raised so full code blocks don't get cut off
)


def _sanitize_book_type(book_type: str) -> str:
    """Prevent path-traversal / unexpected folder names and enforce known modes."""
    return book_type if book_type in VALID_BOOK_TYPES else "coding"


def _db_dir(book_type: str) -> str:
    """
    BUG FIX #1: Each mode (coding / novel / manga) now gets its own persistent
    Chroma directory instead of sharing a single './chroma_db' folder.
    Previously, uploading a new document in ANY mode wiped out every other
    mode's data because they all pointed at the same DB_DIR.
    """
    return f"./chroma_db_{_sanitize_book_type(book_type)}"


def _reset_database(book_type: str):
    """Only wipes the DB belonging to the specific mode being re-uploaded."""
    db_dir = _db_dir(book_type)
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir, ignore_errors=True)
    os.makedirs(db_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM call wrapper with retry/backoff
# ---------------------------------------------------------------------------

def _llm_text(prompt: str, retries: int = 5) -> str:
    """
    BUG FIX #4: Every LLM call now retries with exponential backoff on
    rate-limit (429) errors instead of crashing the whole request. This
    matters a lot for _global_summary(), which can fire off dozens of
    sequential calls on a large book and will hit the free-tier RPM limit.
    """
    last_error = None
    for attempt in range(retries):
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else response

            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)
                return "\n".join(text_parts)

            return str(content)

        except Exception as e:
            last_error = e
            error_str = str(e)
            is_rate_limit = "429" in error_str or "ResourceExhausted" in error_str or "rate" in error_str.lower()

            if is_rate_limit and attempt < retries - 1:
                wait_time = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                time.sleep(wait_time)
                continue
            else:
                raise last_error

    raise last_error


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

def _annotate_pages(docs: List[Document]) -> List[Document]:
    """Inject physical page and line locations into text before embedding."""
    annotated = []
    for idx, doc in enumerate(docs):
        page_number = idx + 1
        lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
        if not lines and doc.page_content.strip():
            lines = [doc.page_content.strip()]

        numbered = [
            f"[Source: PDF Viewer Page {page_number} | Line {line_no}] {line}"
            for line_no, line in enumerate(lines, start=1)
        ]

        metadata = dict(doc.metadata)
        metadata["page_number"] = page_number
        annotated.append(
            Document(page_content="\n".join(numbered), metadata=metadata)
        )
    return annotated


def _get_tesseract_cmd() -> str:
    """
    BUG FIX #2: The old code hardcoded a Windows-only path
    (C:\\Program Files\\Tesseract-OCR\\tesseract.exe), which crashes
    instantly on Mac/Linux/cloud environments. This now:
      1. Checks the TESSERACT_CMD env var (explicit override), then
      2. Looks up 'tesseract' on the system PATH (works on Linux/Mac
         after `apt-get install tesseract-ocr` / `brew install tesseract`), then
      3. Falls back to the Windows default ONLY if actually on Windows.
    """
    env_override = os.environ.get("TESSERACT_CMD")
    if env_override and os.path.exists(env_override):
        return env_override

    found = shutil.which("tesseract")
    if found:
        return found

    windows_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and os.path.exists(windows_default):
        return windows_default

    raise RuntimeError(
        "Tesseract OCR binary not found. Install it first:\n"
        "  Linux:   sudo apt-get install tesseract-ocr\n"
        "  macOS:   brew install tesseract\n"
        "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "Or set the TESSERACT_CMD environment variable to its full path."
    )


def _ocr_manga(file_path: str) -> List[Document]:
    """Render manga pages and extract dialogue using Tesseract OCR."""
    pytesseract.pytesseract.tesseract_cmd = _get_tesseract_cmd()

    pdf = fitz.open(file_path)
    docs = []
    os.makedirs(TEMP_DIR, exist_ok=True)

    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            image_path = os.path.join(TEMP_DIR, f"ocr_page_{page_index + 1}.png")
            pix.save(image_path)

            try:
                text = pytesseract.image_to_string(
                    image_path, lang="eng", config="--psm 6"
                ).strip()

                if text:
                    lines = [
                        f"[Source: PDF Viewer Page {page_index + 1} | Line {line_no}] {line.strip()}"
                        for line_no, line in enumerate(text.splitlines(), start=1)
                        if line.strip()
                    ]
                    docs.append(
                        Document(
                            page_content="\n".join(lines),
                            metadata={
                                "page_number": page_index + 1,
                                "source": file_path,
                                "book_type": "manga",
                            },
                        )
                    )
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)
    finally:
        pdf.close()

    return docs


def process_and_store_document(file_path: str, book_type: str = "coding") -> str:
    """Extract, annotate, chunk and store the uploaded document."""
    book_type = _sanitize_book_type(book_type)
    _reset_database(book_type)

    if book_type == "manga":
        docs = _ocr_manga(file_path)
        if not docs:
            raise ValueError("OCR could not extract readable text from the manga PDF.")
    else:
        loader = PyMuPDFLoader(file_path)
        docs = _annotate_pages(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=_db_dir(book_type),
    )

    return "Success"


# ---------------------------------------------------------------------------
# Question classification
# ---------------------------------------------------------------------------

def _history_text(history: List[Dict[str, Any]], limit: int = 40) -> str:
    if not history:
        return "No previous conversation."
    return "\n".join(
        f"{item.get('role', 'user').upper()}: {item.get('content', '')}"
        for item in history[-limit:]
    )


# BUG FIX #3: Old term list only matched the literal word "book", so
# "What is this document about?" (from the QA screenshot) never triggered
# the global-summary path and instead fell back to a narrow k=4 retrieval
# that couldn't answer a broad question. Now uses regex patterns covering
# "document" / "pdf" / "text" / "manga" / "comic" and catches any
# "summar*" variant (summarize, summary, summarise, summarizing, etc.)
_GLOBAL_PATTERNS = [
    r"entire (book|document)",
    r"whole (book|document)",
    r"summar\w*",
    r"\boverall\b",
    r"\boverview\b",
    r"main themes?",
    r"main topics?",
    r"main idea",
    r"key ideas?",
    r"across the (book|document)",
    r"throughout the (book|document)",
    r"best problem",
    r"most important",
    r"what (is|'s) (this|the) (book|document|text|pdf|story|manga|comic)\s*about",
    r"what is this about",
]


def _is_global_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(pattern, q) for pattern in _GLOBAL_PATTERNS)


def _extract_requested_pages(question: str) -> List[int]:
    """
    BUG FIX #6: The old version only matched literal 'page <digits>' and
    only ever returned ONE page number via re.search(). This broke on:
      - "page number 118"        (word "number" in between)
      - "page 68,145,167"        (comma-separated list -> only got 68)
      - "summary from 138-145"   (a range, with no word "page" at all)
    Now returns a sorted list of ALL page numbers referenced, handling
    ranges (138-145), comma lists (68,145,167), and the word "number".
    """
    q = question.lower()
    pages = set()

    # Ranges anywhere in the text, e.g. "138-145" or "68–145"
    for m in re.finditer(r"\b(\d{1,4})\s*[-–]\s*(\d{1,4})\b", q):
        start, end = int(m.group(1)), int(m.group(2))
        if 0 < end - start < 300:  # sanity guard against false positives
            pages.update(range(min(start, end), max(start, end) + 1))

    # "page(s) [number] 118" / "page 68,145,167" style lists
    match = re.search(r"pages?\s*(?:number)?\s*((?:\d{1,4}[\s,]*)+)", q)
    if match:
        for part in re.split(r"[,\s]+", match.group(1).strip()):
            if part.isdigit():
                pages.add(int(part))

    return sorted(pages)


def _wants_code(question: str) -> bool:
    """Detects requests that expect an actual code block in the answer."""
    q = question.lower()
    triggers = [
        "write code", "write a code", "code of it", "code for",
        "give me code", "give the code", "implement", "write the code",
    ]
    return any(t in q for t in triggers)


# BUG FIX #5: "How many questions have I asked?" used to be answered by
# asking the LLM to count from raw history text, which LLMs do unreliably
# (the screenshot showed it answering "one" when several had been asked).
# Counting is now done deterministically in Python.
_COUNT_PATTERNS = [
    r"how many questions",
    r"how many times have i asked",
    r"number of questions",
]


def _is_count_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _COUNT_PATTERNS)


_MEMORY_TERMS = [
    "what did i ask",
    "what was my question",
    "first question",
    "last question",
    "previous question",
    "what questions did i ask",
    "what did i just ask",
    "what did i say",
    "what have i asked",
    "in this conversation",
    "in this chat",
]


def _is_memory_question(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _MEMORY_TERMS)


# ---------------------------------------------------------------------------
# Global (whole-book) summarization — map/reduce
# ---------------------------------------------------------------------------

def _global_summary(vector_db: Chroma, question: str) -> str:
    data = vector_db.get(include=["documents", "metadatas"])
    documents = data.get("documents", [])

    if not documents:
        return "I cannot find any content in the uploaded document."

    batch_size = 20
    summaries = []

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        context = "\n\n".join(batch)
        summaries.append(
            _llm_text(
                "Summarize the following consecutive sections of a textbook. "
                "Keep important facts, concepts, examples and conclusions. "
                "Do not invent information.\n\n" + context
            )
        )

    while len(summaries) > 1:
        next_summaries = []
        for start in range(0, len(summaries), 8):
            group = summaries[start:start + 8]
            next_summaries.append(
                _llm_text(
                    "Combine these partial textbook summaries into one "
                    "accurate summary. Preserve important details and do "
                    "not add facts not present in the summaries.\n\n"
                    + "\n\n".join(group)
                )
            )
        summaries = next_summaries

    code_instruction = (
        "\nThe user explicitly asked for CODE, not just a description. "
        "Include a complete, correctly formatted code block (proper "
        "syntax highlighting, e.g. ```python). Write real, runnable code "
        "implementing the problem/solution being discussed — do not just "
        "describe it in prose.\n"
        if _wants_code(question) else ""
    )

    return _llm_text(
        "Answer the user's global question using this complete-book "
        "summary. Cover the whole document, but be as detailed as the "
        "question requires. Do not hallucinate.\n"
        f"{code_instruction}\n"
        f"User question: {question}\n\n"
        f"Book summary:\n{summaries[0]}"
    )


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def query_rag_system(
    user_message: str,
    book_type: str = "coding",
    history: List[Dict[str, Any]] | None = None,
) -> str:
    book_type = _sanitize_book_type(book_type)
    history = history or []

    # -----------------------------------------------------------------
    # 1. Deterministic "how many questions" counting (no LLM needed)
    # -----------------------------------------------------------------
    if _is_count_question(user_message):
        count = sum(1 for item in history if item.get("role") == "user")
        return f"You have asked me {count} question{'s' if count != 1 else ''} so far."

    # -----------------------------------------------------------------
    # 2. Other conversation-memory questions
    # -----------------------------------------------------------------
    if _is_memory_question(user_message):
        history_text = _history_text(history)
        prompt = f"""You are answering a question about the conversation history.

Use ONLY the conversation history below.
Do NOT use the textbook.
Do NOT say that the answer is not in the textbook.

Conversation history:
{history_text}

Current question:
{user_message}

Answer accurately and directly:"""
        return _llm_text(prompt)

    # -----------------------------------------------------------------
    # 3. Make sure a document has been uploaded for THIS mode
    # -----------------------------------------------------------------
    db_dir = _db_dir(book_type)
    if not os.path.exists(db_dir) or not os.listdir(db_dir):
        return f"Please upload a {book_type} document first."

    vector_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

    # -----------------------------------------------------------------
    # 4. Explicit page(s)/range requests — CHECKED BEFORE the global
    #    check, so "write a summary from 138-145" pulls only those
    #    pages instead of triggering a whole-book summary (the old
    #    "summar*" pattern would have caught this and ignored the
    #    page numbers entirely).
    # -----------------------------------------------------------------
    requested_pages = _extract_requested_pages(user_message)
    page_docs = []

    if requested_pages:
        try:
            data = vector_db.get(
                where={"page_number": {"$in": requested_pages}},
                include=["documents", "metadatas"],
            )
            page_docs = [
                Document(page_content=d, metadata=m or {})
                for d, m in zip(data.get("documents", []), data.get("metadatas", []))
            ]
        except Exception:
            page_docs = []

    if page_docs:
        # Sort retrieved chunks by page number so the answer reads in order
        page_docs.sort(key=lambda d: d.metadata.get("page_number", 0))
        context = "\n\n".join(doc.page_content for doc in page_docs)
        history_text = _history_text(history)

        code_instruction = (
            "\nThe user explicitly asked for CODE. Include a complete, "
            "correctly formatted code block (proper syntax highlighting, "
            "e.g. ```python). Do not just describe the solution in prose — "
            "provide runnable code.\n"
            if _wants_code(user_message) else ""
        )

        prompt = f"""You are a precise document assistant.

Answer the user's question using ONLY:
1. The supplied document context (from the specific page(s) requested: {requested_pages}).
2. The conversation history when it is relevant (e.g. comparisons to something discussed earlier).

If the requested pages contain no relevant content, say so honestly.
Never invent facts.
{code_instruction}
Conversation history:
{history_text}

Document context (pages {requested_pages}):
{context}

Current question:
{user_message}

Answer:"""
        return _llm_text(prompt)

    # -----------------------------------------------------------------
    # 5. Global / whole-book questions (only reached if no explicit
    #    page numbers were found in the question)
    # -----------------------------------------------------------------
    if _is_global_question(user_message):
        return _global_summary(vector_db, user_message)

    # -----------------------------------------------------------------
    # 6. Normal semantic RAG retrieval
    # -----------------------------------------------------------------
    retriever = vector_db.as_retriever(search_kwargs={"k": 8})
    docs = retriever.invoke(user_message)
    context = "\n\n".join(doc.page_content for doc in docs)

    # -----------------------------------------------------------------
    # 7. Normal document question
    # -----------------------------------------------------------------
    history_text = _history_text(history)

    code_instruction = (
        "\nThe user explicitly asked for CODE. Include a complete, "
        "correctly formatted code block (proper syntax highlighting, "
        "e.g. ```python). Do not just describe the solution in prose — "
        "provide runnable code.\n"
        if _wants_code(user_message) else ""
    )

    prompt = f"""You are a precise document assistant.

Answer the user's question using ONLY:
1. The supplied document context.
2. The conversation history when it is relevant.

If the answer is not present in the document context, say:
"I cannot find the answer to this in the textbook."

Never invent facts.

For page/location questions, respect the PDF Viewer Page and
Line labels exactly.
{code_instruction}
Conversation history:
{history_text}

Document context:
{context}

Current question:
{user_message}

Answer:"""

    return _llm_text(prompt)