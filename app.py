from fastapi import FastAPI
from pydantic import BaseModel

from semantic_cache import (
    semantic_search,
    semantic_cache,
    hit_count,
    miss_count,
    cache_path,
)

import os
import pickle


app = FastAPI()


# =========================
# REQUEST MODEL
# =========================


class QueryRequest(BaseModel):
    query: str


# =========================
# POST /query
# =========================


@app.post("/query")
def query_endpoint(request: QueryRequest):

    response = semantic_search(request.query)

    return {"query": request.query, **response}


# =========================
# GET /cache/stats
# =========================


@app.get("/cache/stats")
def cache_stats():

    total_entries = sum(len(v) for v in semantic_cache.values())

    total_requests = hit_count + miss_count

    hit_rate = hit_count / total_requests if total_requests > 0 else 0

    return {
        "total_entries": total_entries,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": round(hit_rate, 3),
    }


# =========================
# DELETE /cache
# =========================


@app.delete("/cache")
def clear_cache():

    semantic_cache.clear()

    if os.path.exists(cache_path):
        os.remove(cache_path)

    return {"message": "Cache cleared successfully"}
