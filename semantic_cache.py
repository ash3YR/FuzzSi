import numpy as np
import faiss
import joblib
import pickle
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# LOAD ALL MODELS
# =========================
# Sentence embedding model used for semantic query encoding

model = SentenceTransformer("all-MiniLM-L6-v2")


# PCA and GMM saved from Part 2 clustering
# Required so incoming queries enter same semantic space

pca = joblib.load("generated_docs/pca_model.pkl")
gmm = joblib.load("generated_docs/gmm_model.pkl")


# FAISS index built in Part 1
# Used when cache miss occurs

index = faiss.read_index("vector_store/faiss_index.index")


# =========================
# LOAD CACHE FROM DISK
# =========================
# Persistent cache survives script restarts

cache_path = "generated_docs/semantic_cache.pkl"

if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        semantic_cache = pickle.load(f)
else:
    semantic_cache = {}


# =========================
# CACHE STATS
# =========================

hit_count = 0
miss_count = 0


# =========================
# FIND QUERY CLUSTER
# =========================
# Route query into semantic cluster using same PCA + GMM pipeline


def get_query_cluster(query_embedding):

    reduced = pca.transform(query_embedding.reshape(1, -1))

    probs = gmm.predict_proba(reduced)

    cluster = np.argmax(probs)

    return cluster


# =========================
# SEARCH CACHE
# =========================
# Search only inside relevant semantic cluster


def search_cache(query, threshold=0.85):

    global hit_count

    query_embedding = model.encode([query]).astype("float32")

    cluster = get_query_cluster(query_embedding[0])

    print("Query Cluster:", cluster)

    if cluster not in semantic_cache:
        return None

    entries = semantic_cache[cluster]

    for entry in entries:
        sim = cosine_similarity(query_embedding, entry["embedding"].reshape(1, -1))[0][
            0
        ]

        if sim >= threshold:
            hit_count += 1

            print("CACHE HIT")

            return {
                "cache_hit": True,
                "matched_query": entry["query"],
                "similarity_score": float(sim),
                "result": entry["result"],
                "dominant_cluster": int(cluster),
            }

    return None


# =========================
# FAISS SEARCH
# =========================
# Executed only when semantic cache misses


def search_faiss(query):

    query_embedding = model.encode([query]).astype("float32")

    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, 5)

    return I[0].tolist()


# =========================
# INSERT INTO CACHE
# =========================
# Store query inside semantic cluster and persist to disk


def insert_cache(query, result):

    query_embedding = model.encode([query]).astype("float32")

    cluster = get_query_cluster(query_embedding[0])

    entry = {"query": query, "embedding": query_embedding[0], "result": result}

    if cluster not in semantic_cache:
        semantic_cache[cluster] = []

    semantic_cache[cluster].append(entry)

    with open(cache_path, "wb") as f:
        pickle.dump(semantic_cache, f)


# =========================
# MAIN SEMANTIC SEARCH
# =========================
# cache first → FAISS fallback → store result


def semantic_search(query, threshold=0.85):

    global miss_count

    cached = search_cache(query, threshold)

    if cached is not None:
        return cached

    miss_count += 1

    result = search_faiss(query)

    insert_cache(query, result)

    cluster = get_query_cluster(model.encode([query]).astype("float32")[0])

    print("CACHE MISS → STORED")

    return {
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": int(cluster),
    }


# =========================
# TEST RUN
# =========================

if __name__ == "__main__":
    q1 = "kidney stone treatment"
    result1 = semantic_search(q1)

    print("\nFirst Query Result:", result1)

    q2 = "how to treat kidney stones"
    result2 = semantic_search(q2)

    print("\nSecond Query Result:", result2)
