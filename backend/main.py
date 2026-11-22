from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import shutil
import os

from rag_engine import (
    process_and_store_document,
    query_rag_system,
)

app = FastAPI(title="Zenvyrolabs Task 3 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    message: str
    book_type: str = "coding"
    history: List[Dict[str, Any]] = Field(default_factory=list)


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    book_type: str = Form("coding"),
):
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)

        file_path = os.path.join(
            TEMP_DIR,
            os.path.basename(file.filename),
        )

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        process_and_store_document(file_path, book_type)

        return {
            "message": (
                f"Successfully processed {file.filename} "
                f"as a {book_type} document!"
            )
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        answer = query_rag_system(
            request.message,
            request.book_type,
            request.history,
        )
        return {"answer": answer}

    except Exception as e:
        return {
            "error": f"Failed to query system: {str(e)}"
        }


frontend_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "frontend",
)

if os.path.exists(frontend_path):
    app.mount(
        "/",
        StaticFiles(
            directory=frontend_path,
            html=True,
        ),
        name="frontend",
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
    )