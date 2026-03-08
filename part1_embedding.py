import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# STEP 1: LOAD UCI DATASET


print("Starting loading..")

base_path = "data/20_newsgroups"

documents = []
labels = []

print("c1")  # remove this later

for category in os.listdir(base_path):
    category_path = os.path.join(base_path, category)

    print("c2")  # remove this later

    if os.path.isdir(category_path):
        for filename in os.listdir(category_path):
            if filename.startswith("."):
                continue

            file_path = os.path.join(category_path, filename)

            with open(file_path, "r", encoding="latin1") as f:
                text = f.read()

                documents.append(text)
                labels.append(category)

print("Total documents loaded:", len(documents))


# STEP 2: CLEAN TEXT


# def clean_text(text):
#     lines = text.split("\n")
#     cleaned_lines = []

#     for line in lines:
#         # Remove quoted replies because they repeat previous discussions
#         if line.startswith(">"):
#             continue

#         # Remove metadata headers because they do not represent semantic meaning
#         if line.startswith("From:"):
#             continue
#         if line.startswith("Subject:"):
#             continue
#         if line.startswith("Organization:"):
#             continue
#         if line.startswith("Lines:"):
#             continue

#         cleaned_lines.append(line)

#     text = " ".join(cleaned_lines)

#     # Remove email addresses
#     text = re.sub(r"\S+@\S+", " ", text)

#     # Remove multiple spaces
#     text = re.sub(r"\s+", " ", text)

#     return text.strip()


def clean_text(text):
    lines = text.split("\n")

    cleaned_lines = []

    header_prefixes = [
        "From:",
        "Subject:",
        "Organization:",
        "Lines:",
        "Path:",
        "Newsgroups:",
        "Date:",
        "Message-ID:",
        "References:",
        "Sender:",
        "Reply-To:",
        "Distribution:",
        "Xref:",
        "NNTP-Posting-Host:",
        "Keywords:",
        "Summary:",
        "Originator:",
    ]

    for line in lines:
        if line.startswith(">"):
            continue

        if any(line.startswith(prefix) for prefix in header_prefixes):
            continue

        cleaned_lines.append(line)

    text = " ".join(cleaned_lines)

    text = re.sub(r"\S+@\S+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


cleaned_docs = [clean_text(doc) for doc in tqdm(documents)]


# STEP 3: SAVE CLEANED METADATA

df = pd.DataFrame({"text": cleaned_docs, "label": labels})

df.to_csv("embeddings/cleaned_documents.csv", index=False)

print("Cleaned metadata saved.")


# STEP 4: LOAD EMBEDDING MODEL

# all-MiniLM-L6-v2 chosen because of it's lightweight, fast, strong semantic quality for medium corpus size

model = SentenceTransformer("all-MiniLM-L6-v2")


# STEP 5: GENERATE EMBEDDINGS

embeddings = model.encode(cleaned_docs, batch_size=64, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)


# STEP 6: SAVE EMBEDDINGS

np.save("embeddings/document_embeddings.npy", embeddings)

print("Embeddings saved.")


# STEP 7: NORMALIZE FOR COSINE SEARCH

faiss.normalize_L2(embeddings)


# STEP 8: BUILD FAISS VECTOR DATABASE

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS index built.")


# STEP 9: SAVE VECTOR DATABASE

faiss.write_index(index, "vector_store/faiss_index.index")

print("Vector database saved.")


# STEP 10: SAVE DOCUMENT INDEX MAP

mapping = pd.DataFrame({"doc_id": range(len(cleaned_docs)), "label": labels})

mapping.to_csv("vector_store/doc_mapping.csv", index=False)

print("Document mapping saved.")
