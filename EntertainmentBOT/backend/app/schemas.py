"""Request/response models for the public API (handy for mobile/other clients)."""
from typing import Any, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    # Answers from the clarifying-question widget, keyed by field name.
    answers: Optional[dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    rating: str  # "satisfied" | "dissatisfied"
    # 1-5 stars, given after the user taps "satisfied".
    app_rating: Optional[int] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    comment: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    titles: list[dict] = []


class UnlockRequest(BaseModel):
    passphrase: str


class SessionSummary(BaseModel):
    id: str
    title: str
    updated_at: float
    query_count: int = 0
    user_name: Optional[str] = None
