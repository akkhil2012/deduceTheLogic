# dedup_legal_docs.py
# --------------------------------------------------
# Goal: Detect duplicates / near-duplicates in legal firm evidence data
# Approach: TF-IDF vectorization + cosine similarity

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------
# 1. Sample "legal evidence" dataset
# -------------------------
data = {
    "doc_id": [1, 2, 3, 4, 5],
    "content": [
        "This contract is between Company A and Company B signed on Jan 2020.",
        "This contract is between Company A and Company B signed on January 2020.",  # near-duplicate
        "The meeting notes discuss the merger of Company C and Company D.",
        "Confidential: Evidence submitted to the court in Case 1456.",
        "Confidential evidence submitted to the court in case number 1456."  # near-duplicate
    ]
}

df = pd.DataFrame(data)

# -------------------------
# 2. TF-IDF vectorization
# -------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["content"])

# -------------------------
# 3. Cosine similarity
# -------------------------
similarity_matrix = cosine_similarity(tfidf_matrix)

# -------------------------
# 4. Identify duplicates / near-duplicates
# -------------------------
threshold = 0.7  # adjust for strictness
duplicates = []

for i in range(len(df)):
    for j in range(i+1, len(df)):
        sim = similarity_matrix[i, j]
        if sim >= threshold:
            duplicates.append((df.loc[i, "doc_id"], df.loc[j, "doc_id"], sim))

# -------------------------
# 5. Output results
# -------------------------
print("Potential Duplicates / Near-Duplicates:")
for d in duplicates:
    print(f"Doc {d[0]} <-> Doc {d[1]} | Similarity: {d[2]:.3f}")

