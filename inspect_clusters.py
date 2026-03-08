import numpy as np
import pandas as pd

# LOAD FUZZY OUTPUT

probabilities = np.load("clusters_probabilities.npy")

df = pd.read_csv("embeddings/cleaned_documents.csv")

num_clusters = probabilities.shape[1]

print("Total clusters:", num_clusters)


# FOR EACH CLUSTER

for cluster_id in range(num_clusters):
    print("\n" + "=" * 60)
    print(f"CLUSTER {cluster_id}")
    print("=" * 60)

    # highest probability docs for this cluster
    top_indices = probabilities[:, cluster_id].argsort()[-5:][::-1]

    for idx in top_indices:
        score = probabilities[idx, cluster_id]

        print(f"\nDocument Index: {idx}")
        print(f"Membership Score: {score:.3f}")

        print("Text Preview:")
        print(df.iloc[idx]["text"][:400])

        print("-" * 50)
