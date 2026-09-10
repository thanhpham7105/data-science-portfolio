import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# D. DATA PREPARATION
# D1. Load and inspect data
churn_df = pd.read_csv("churn_clean.csv")

# D2. Select continuous variables for k-means
features = [
    "Tenure",
    "MonthlyCharge",
    "Bandwidth_GB_Year",
    "Outage_sec_perweek",
    "Age",
    "Income",
    "Item1", "Item2", "Item3", "Item4",
    "Item5", "Item6", "Item7", "Item8"
]
data = churn_df[features].copy()

# D3. Handle missing values
data_filled = data.fillna(data.median(numeric_only=True))

# D4. Standardize features and save cleaned dataset
scaler = StandardScaler()
scaled_array = scaler.fit_transform(data_filled)
scaled_df = pd.DataFrame(scaled_array, columns=features)

# Save the cleaned, standardized dataset
scaled_df.to_csv("cleaned_cluster_dataset.csv", index=False)

# E. DATA ANALYSIS
# E1. Determine optimal number of clusters (elbow method)
inertias = []
k_values = range(2, 11)
for k in k_values:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(scaled_df)
    inertias.append(kmeans_temp.inertia_)
    
plt.figure(figsize=(7, 5))
plt.plot(k_values, inertias, marker="o")
plt.title("Elbow Method for Selecting k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.tight_layout()
plt.show()

# After looking at the elbow plot in the report, I choose k = 4
optimal_k = 4

# Fit final k-means model with chosen k
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(scaled_df)

# Add cluster labels back to the standardized data
scaled_df["Cluster"] = cluster_labels
clustered_customers = data_filled.copy()
clustered_customers["Cluster"] = cluster_labels

print("\nCluster label counts:")
print(scaled_df["Cluster"].value_counts().sort_index())

# F. SUMMARY OF DATA ANALYSIS
# Silhouette score for cluster quality
sil_score = silhouette_score(scaled_array, cluster_labels)
print(f"\nSilhouette score for k = {optimal_k}: {sil_score:.3f}")

# F1. Visualize clusters with PCA
pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(scaled_array)

clustered_customers["PC1"] = pca_components[:, 0]
clustered_customers["PC2"] = pca_components[:, 1]

plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    clustered_customers["PC1"],
    clustered_customers["PC2"],
    c=clustered_customers["Cluster"],
    cmap="viridis",
    alpha=0.7
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"Customer Segments (k={kmeans_final}) in PCA Space")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("kmeans_clusters_pca_churn.png", dpi=150)
plt.show()

# F2. Cluster results summary
print("\nCluster mean values of the original features:")
cluster_summary = clustered_customers.groupby("Cluster")[features].mean()
print(cluster_summary)
cluster_summary.to_csv("cluster_summary_kmeans_churn.csv")