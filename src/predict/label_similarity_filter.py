import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("../dataset/merged_with_angles.csv")

# 2. Select feature columns (coordinates + angles)
coordinate_cols = [col for col in df.columns if col.startswith(('lx', 'ly', 'rx', 'ry'))]
angle_cols = [col for col in df.columns if col.startswith(('angle_l_', 'angle_r_'))]
feature_cols = coordinate_cols + angle_cols

# 3. Compute mean vector per label
label_vectors = {}
grouped = df.groupby("label")

for label, group in grouped:
    features = group[feature_cols].values
    mean_vector = features.mean(axis=0)
    label_vectors[label] = mean_vector

# 4. Cosine similarity between label vectors
labels = list(label_vectors.keys())
vectors = np.stack([label_vectors[label] for label in labels])
sim_matrix = cosine_similarity(vectors)

# 5. Compute mean similarity per label (excluding self)
mean_similarities = []
for i, row in enumerate(sim_matrix):
    mean_sim = (np.sum(row) - 1) / (len(row) - 1)
    mean_similarities.append((labels[i], mean_sim))

# 6. Select 30 least similar labels (most distinct)
mean_similarities.sort(key=lambda x: x[1])
selected_labels = [label for label, _ in mean_similarities[:30]]

# 7. Print result
print("Top 30 most distinct labels (lowest similarity):")
for label in selected_labels:
    print("-", label)

# Plot 1: Histogram of mean similarity distribution
_, avg_similarities = zip(*mean_similarities)
plt.figure(figsize=(10, 6))
plt.hist(avg_similarities, bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Mean Similarity")
plt.xlabel("Mean Similarity")
plt.ylabel("Number of Labels")
plt.grid(True)
plt.show()

# Plot 2: Scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(range(len(avg_similarities)), avg_similarities, c='orange', s=30)
plt.title("Mean Similarity per Label")
plt.xlabel("Label Index")
plt.ylabel("Mean Similarity")
plt.grid(True)
plt.show()