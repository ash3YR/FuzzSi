# FuzzSi — Semantic Retrieval, Fuzzy Clustering, and Semantic Cache API

FuzzSi is a three-part semantic systems project designed to demonstrate:

* semantic document embedding and retrieval
* fuzzy clustering over text corpora
* semantic query caching through a FastAPI service

The project combines offline semantic preprocessing with an online cache-enabled retrieval API.

---

# Project Structure

```text
FuzzSi/
├── app.py
├── semantic_cache.py
├── part1_embedding.py
├── part2_clustering.py
├── inspect_clusters.py
├── requirements.txt
├── README.md
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
│
├── embeddings/
│   ├── cleaned_documents.csv
│   └── document_embeddings.npy
│
├── vector_store/
│   ├── faiss_index.index
│   └── doc_mapping.csv
│
├── generated_docs/
│   ├── clustered_documents.csv
│   ├── uncertain_documents.csv
│   ├── boundary_documents.csv
│   ├── clusters_probabilities.npy
│   ├── clusters_bic.png
│   ├── gmm_model.pkl
│   └── pca_model.pkl
│
├── tests/
│   └── test_cache.py
```

---

# Part 1 — Embedding Generation and Vector Store

This stage prepares semantic document embeddings and stores them in a vector index for retrieval.

## Objective

Convert cleaned text documents into dense semantic embeddings and index them using FAISS.

## Files

* `part1_embedding.py`
* `embeddings/document_embeddings.npy`
* `vector_store/faiss_index.index`
* `vector_store/doc_mapping.csv`

## Pipeline

1. Load cleaned document corpus
2. Generate sentence embeddings
3. Save embedding matrix
4. Build FAISS index
5. Save document mapping

## Run

```bash
python part1_embedding.py
```

## Outputs

* `document_embeddings.npy`
* `faiss_index.index`
* `doc_mapping.csv`

---

# Part 2 — Fuzzy Clustering

This stage discovers latent semantic structure using probabilistic clustering.

## Objective

Assign soft cluster memberships to documents instead of hard labels.

## Files

* `part2_clustering.py`
* `inspect_clusters.py`

## Method Used

* PCA for dimensionality reduction
* Gaussian Mixture Model (GMM)
* BIC-based cluster count selection
* Temperature smoothing for softer memberships

## Run

```bash
python part2_clustering.py
```

## Generated Outputs

### `clustered_documents.csv`

Contains:

* dominant cluster assignment
* entropy score

### `uncertain_documents.csv`

Documents with highest semantic uncertainty.

### `boundary_documents.csv`

Documents lying between clusters.

### `clusters_probabilities.npy`

Full fuzzy membership matrix.

### `clusters_bic.png`

BIC curve for cluster selection.

## Interpretation

Each document may partially belong to multiple semantic regions.

Example:

* cluster A = 0.54
* cluster B = 0.31

This captures semantic overlap better than hard clustering.

---

# Part 3 — Semantic Cache API

This stage exposes semantic retrieval through a FastAPI service.

## Objective

Reuse previous answers when semantically similar queries are asked.

## Files

* `app.py`
* `semantic_cache.py`

## Features

* semantic similarity matching
* cache hit / miss detection
* similarity threshold reuse
* cache statistics
* cache reset endpoint

## Run API

```bash
uvicorn app:app --reload
```

This satisfies the single-command startup requirement.

---

# API Endpoints

## POST `/query`

Input:

```json
{
  "query": "kidney stone treatment"
}
```

Example response:

```json
{
  "answer": "...",
  "cache_hit": false
}
```

---

## GET `/cache/stats`

Returns:

```json
{
  "hit_count": 1,
  "miss_count": 1,
  "hit_rate": 0.5
}
```

---

## DELETE `/cache`

Clears cache.

---

# Semantic Cache Behavior

Example:

* First query: `"kidney stone treatment"` → cache miss
* Second query: `"how to treat kidney stones"` → cache hit

The second query reuses the previous answer through semantic similarity.

---

# Environment Setup

Create virtual environment:

```bash
python -m venv venv
```

Activate:

## Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dependencies

Freeze environment:

```bash
pip freeze > requirements.txt
```

---

# Testing

## Cache test

```bash
python tests/test_cache.py
```

## Cluster inspection

```bash
python inspect_clusters.py
```

---

# Docker Support (Bonus)

The FastAPI service is containerized for deployment.

## Build image

```bash
docker build -t fuzzsi .
```

## Run container

```bash
docker run -p 8000:8000 fuzzsi
```

## Or using Docker Compose

```bash
docker compose up
```

The container starts the FastAPI server on port **8000**.

---

# Deployment Note

Only the FastAPI service is containerized because embedding generation and clustering are offline preprocessing stages, while semantic cache serves as the deployable runtime component.
