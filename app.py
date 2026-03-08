from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

from semantic_cache import semantic_search, semantic_cache, cache_path

import semantic_cache as cache_module


# =========================
# FASTAPI APP
# =========================

app = FastAPI()


# =========================
# ENABLE CORS
# =========================
# Safe to keep for local testing / Postman / browser

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REQUEST MODEL
# =========================


class QueryRequest(BaseModel):
    query: str


# =========================
# POST /query
# =========================
# Semantic search endpoint:
# cache lookup → FAISS fallback → return metadata


@app.post("/query")
def query_endpoint(request: QueryRequest):

    response = semantic_search(request.query)

    return {"query": request.query, **response}


# =========================
# GET /cache/stats
# =========================
# Returns live cache statistics


@app.get("/cache/stats")
def cache_stats():

    total_entries = sum(len(v) for v in semantic_cache.values())

    total_requests = cache_module.hit_count + cache_module.miss_count

    hit_rate = cache_module.hit_count / total_requests if total_requests > 0 else 0

    return {
        "total_entries": total_entries,
        "hit_count": cache_module.hit_count,
        "miss_count": cache_module.miss_count,
        "hit_rate": round(hit_rate, 3),
    }


# =========================
# DELETE /cache
# =========================
# Clears cache + resets stats


@app.delete("/cache")
def clear_cache():

    semantic_cache.clear()

    cache_module.hit_count = 0
    cache_module.miss_count = 0

    if os.path.exists(cache_path):
        os.remove(cache_path)

    return {"message": "Cache cleared successfully"}
