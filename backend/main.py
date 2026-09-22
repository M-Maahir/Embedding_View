"""FastAPI app: embed words or sentences and serve the Three.js frontend."""

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .embeddings import MAX_ITEMS, build_payload, build_sentence_payload

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="3D Word Space")


class EmbedRequest(BaseModel):
    mode: Literal["words", "sentences"] = "words"
    words: list[str] | None = Field(default=None)
    text: str | None = Field(default=None)

    @model_validator(mode="after")
    def require_input(self):
        if self.mode == "sentences":
            if not (self.text and self.text.strip()):
                raise ValueError("Paste text to split into sentences.")
        elif not self.words:
            raise ValueError("Enter at least one word.")
        return self


@app.post("/api/embed")
def embed_items(req: EmbedRequest) -> dict:
    try:
        if req.mode == "sentences":
            return build_sentence_payload(req.text or "")
        return build_payload(req.words or [])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "max_items": MAX_ITEMS}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
