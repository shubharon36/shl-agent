"""Pydantic models for request/response schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation."""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message content")


class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    messages: List[Message] = Field(..., description="Full conversation history")


class Recommendation(BaseModel):
    """A single assessment recommendation."""
    name: str = Field(..., description="Assessment name from the SHL catalog")
    url: str = Field(..., description="Catalog URL for the assessment")
    test_type: str = Field(..., description="Test type code (K, P, A, B, C, S, D, E)")


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    reply: str = Field(..., description="Agent's next reply")
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="List of recommended assessments (empty if still gathering context)"
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete"
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str = "ok"
