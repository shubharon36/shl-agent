"""
FastAPI application for the SHL Assessment Chatbot.
Provides /health and /chat endpoints.
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import ChatRequest, ChatResponse, HealthResponse
from .agent import SHLAgent

# Global agent instance
agent: SHLAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the agent on startup."""
    global agent
    print("Starting SHL Assessment Chatbot...")
    agent = SHLAgent()
    print("Agent ready!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="SHL Assessment Chatbot",
    description="Conversational agent for SHL assessment recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint. Returns 200 with status ok."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a conversation and return the agent's response.
    
    The API is stateless — every call carries the full conversation history.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet")

    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    # Validate that the last message is from a user
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from the user")

    response = await agent.chat(request.messages)
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
