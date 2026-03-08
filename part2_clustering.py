import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# =========================
# STEP 1: LOAD EMBEDDINGS
# =========================

embeddings = np.load("embeddings/document_embeddings.npy")

print("Embeddings shape:", embeddings.shape)


# =========================
# STEP 2: PCA REDUCTION
# =========================
# PCA reduces dimensionality before GMM because
# Gaussian models become unstable in very high-dimensional space.

pca = PCA(n_components=30, random_state=42)

reduced_embeddings = pca.fit_transform(embeddings)

print("Reduced shape:", reduced_embeddings.shape)


# =========================
# STEP 3: FIND BEST CLUSTER COUNT USING BIC
# =========================
# BIC balances model fit against complexity.
# Lower BIC indicates better cluster structure without over-fragmentation.

cluster_range = range(5, 15)

bic_scores = []

for k in cluster_range:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)

    gmm.fit(reduced_embeddings)

    bic = gmm.bic(reduced_embeddings)

    bic_scores.append(bic)

    print(f"Clusters {k}: BIC = {bic}")


# =========================
# STEP 4: PLOT BIC
# =========================

plt.plot(cluster_range, bic_scores)
plt.xlabel("Number of Clusters")
plt.ylabel("BIC Score")
plt.title("Choosing Optimal Cluster Count")
plt.savefig("clusters_bic.png")
plt.show()


# =========================
# STEP 5: CHOOSE BEST K
# =========================

best_k = cluster_range[np.argmin(bic_scores)]

print("Best cluster count:", best_k)


# =========================
# STEP 6: FINAL GMM MODEL
# =========================
# Full covariance allows softer overlap between semantic regions.

gmm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42)

gmm.fit(reduced_embeddings)


# =========================
# STEP 7: FUZZY MEMBERSHIPS
# =========================

probabilities = gmm.predict_proba(reduced_embeddings)


# =========================
# STEP 8: TEMPERATURE SMOOTHING
# =========================
# GMM can become overly confident on strongly separable text corpora.
# Temperature smoothing softens probabilities to reveal semantic overlap.

temperature = 2.5

probabilities = probabilities ** (1 / temperature)

probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

print("Probability matrix shape:", probabilities.shape)


# =========================
# STEP 9: SAVE PROBABILITIES
# =========================

np.save("generated_docs/clusters_probabilities.npy", probabilities)


# =========================
# STEP 10: DOMINANT CLUSTER
# =========================

dominant_cluster = probabilities.argmax(axis=1)


# =========================
# STEP 11: ENTROPY (UNCERTAINTY)
# =========================
# High entropy = document lies across semantic boundaries.

entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)


# =========================
# STEP 12: SAVE FULL CLUSTER OUTPUT
# =========================

df = pd.read_csv("embeddings/cleaned_documents.csv")

df["dominant_cluster"] = dominant_cluster
df["entropy"] = entropy

df.to_csv("generated_docs/clustered_documents.csv", index=False)

print("Cluster assignments saved.")


# =========================
# STEP 13: SAVE TOP UNCERTAIN DOCS
# =========================

top_uncertain = np.argsort(entropy)[-20:]

uncertain_docs = df.iloc[top_uncertain]

uncertain_docs.to_csv("generated_docs/uncertain_documents.csv", index=False)

print("Uncertain documents saved.")


# =========================
# STEP 14: SAVE BOUNDARY DOCS
# =========================
# Small difference between top two memberships = boundary case

sorted_probs = np.sort(probabilities, axis=1)

boundary_score = sorted_probs[:, -1] - sorted_probs[:, -2]

boundary_indices = np.argsort(boundary_score)[:20]

boundary_docs = df.iloc[boundary_indices]

boundary_docs.to_csv("generated_docs/boundary_documents.csv", index=False)

print("Boundary documents saved.")


# ADDITIONAL STEP ADDED WHILE I WAS CODING FOR CACHING


# =========================
# STEP 15: SAVE PCA + GMM MODELS
# =========================
# These are needed later so incoming queries can be projected
# into the same semantic cluster space during cache lookup.

import joblib

joblib.dump(pca, "generated_docs/pca_model.pkl")
joblib.dump(gmm, "generated_docs/gmm_model.pkl")

print("PCA and GMM models saved.")
