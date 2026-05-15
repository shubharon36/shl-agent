# SHL Assessment Chatbot — Approach Document

## Architecture Overview

The system is a stateless FastAPI service with two endpoints: `GET /health` and `POST /chat`. Each `/chat` call receives the full conversation history, retrieves relevant catalog assessments via semantic search, builds a grounded prompt, and calls Google Gemini 2.0 Flash to generate a structured response. All recommended assessments are post-validated against the catalog to prevent hallucinated names or URLs.

```
User Message → Extract Search Queries → FAISS Semantic Search (top-30) →
Build System Prompt with Catalog Context → Gemini 2.0 Flash → Parse JSON →
Validate Recommendations Against Catalog → Return Response
```

## Design Choices

**LLM: Gemini 2.0 Flash** — Chosen for its free tier, fast inference (<5s), strong instruction following, and native JSON output mode (`response_mime_type="application/json"`). Low temperature (0.2) ensures deterministic, grounded responses.

**Retrieval: sentence-transformers + FAISS** — Each of the 377 assessments is embedded using `all-MiniLM-L6-v2` (384-dim) with a rich text combining name, description, categories, job levels, and languages. On each call, we extract the full conversation context and the latest user message as separate queries, retrieve top-25 from each, deduplicate, and pass the top-30 as grounded context to the LLM. This hybrid (full-context + focused) retrieval strategy ensures both cumulative understanding and responsiveness to the latest turn.

**Post-validation** — Every recommendation returned by the LLM is looked up in the catalog by exact name, URL, and fuzzy name match. Only verified assessments with their canonical names and URLs are returned. This eliminates hallucination at the output layer.

## Prompt Design

The system prompt encodes four explicit behavioral rules: (1) **Clarify** vague queries before recommending, (2) **Recommend** 1-10 assessments with grounded data when sufficient context exists, (3) **Refine** the shortlist on constraint changes without restarting, (4) **Compare** assessments using only catalog data. Strict scope guardrails prevent off-topic responses, legal advice, and prompt injection. The catalog context (top-30 retrieved assessments with full metadata) is injected directly into the system prompt, ensuring the LLM can only reference real products.

## Evaluation Approach

Development was driven by the 10 public conversation traces (C1–C10). Key metrics tracked:
- **Schema compliance**: Every response validated for required fields, URL format, and catalog membership.
- **Behavioral probes**: Tested vague queries (no immediate recommendations), off-topic refusal (salary questions), refinement (add/remove tests mid-conversation), and comparison (grounded differences).
- **Recall@10**: Measured against expected shortlists from the 10 public traces.

## What Didn't Work

- **Full catalog in prompt**: Passing all 377 assessments as context exceeded token limits and slowed responses. Semantic retrieval (top-30) was the necessary solution.
- **Higher temperature**: Temperature >0.5 caused the LLM to occasionally invent assessment names not in the catalog. Lowering to 0.2 with post-validation solved this.
- **Single-query retrieval**: Using only the latest user message for search missed cumulative context (e.g., role + seniority mentioned across turns). Using full conversation context as a composite query improved recall.

## AI Tools Used

Agentic coding assistant (Gemini-based) was used for rapid prototyping of the FastAPI service structure, prompt engineering iteration, and test script generation. All architectural decisions, prompt design, and evaluation were human-directed.

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| LLM | Gemini 2.0 Flash | Free tier, fast, JSON mode |
| Embeddings | all-MiniLM-L6-v2 | Small, fast, good quality |
| Vector Store | FAISS (in-memory) | No infrastructure needed |
| API | FastAPI + Uvicorn | Async, fast, auto-docs |
| Deployment | Render (Docker) | Free tier, auto-deploy |
