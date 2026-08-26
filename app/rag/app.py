"""Standalone FastAPI entry point for the optional Evidence RAG layer.

Run with:
    uvicorn app.rag.app:app --reload

Keeping this service separate from the frozen decision API prevents a new RAG
feature from silently changing the published Router/S5 operating profile.
"""

from fastapi import FastAPI

from app.rag.api import router

app = FastAPI(
    title="DeFactoRule Evidence RAG",
    version="0.1.0",
    description=(
        "Temporal-eligible hybrid precedent retrieval with provenance-preserving "
        "evidence IDs and an optional grounded LLM evidence memo."
    ),
)
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "evidence-rag"}
