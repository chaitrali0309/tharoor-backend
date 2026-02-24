from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from agent import ask_agent

app = FastAPI(title="Tharoor Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ───
class HistoryTurn(BaseModel):
    role: str   # "user" or "bot"
    text: str

class AskRequest(BaseModel):
    question: str
    history: Optional[List[HistoryTurn]] = []

class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    history = [{"role": t.role, "text": t.text} for t in req.history] if req.history else []
    answer = ask_agent(req.question, history=history)
    return {"answer": answer}