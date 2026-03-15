
# Step 1: Import Libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Step 2: Data Collection


df = pd.read_csv("app_user_behavior_dataset.csv")

print("Dataset Shape:", df.shape)
df.head()


# Step 3: Data Understanding


print("\nDataset Info\n")
print(df.info())

print("\nStatistical Summary\n")
print(df.describe())


# Step 4: Data Cleaning


# Check missing values
print("\nMissing Values\n")
print(df.isnull().sum())

# Fill numerical missing values
df.fillna(df.median(numeric_only=True), inplace=True)


# Step 5: Feature Selection


features = [
    "sessions_per_week",
    "avg_session_duration_min",
    "daily_active_minutes",
    "engagement_score"
]

X = df[features]

print("\nSelected Features\n")
print(X.head())


# Step 6: Data Scaling


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Step 7: Optimal Cluster Identification
# (Elbow Method)


inertia = []

K = range(2,10)

for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()


# Step 8: Train K-Means Model


kmeans = KMeans(n_clusters=4, random_state=42)

df["cluster"] = kmeans.fit_predict(X_scaled)

print(df.head())


# Step 9: Cluster Profiling


cluster_summary = df.groupby("cluster")[features].mean()

cluster_counts = df["cluster"].value_counts()

print("\nCluster Summary\n")
print(cluster_summary)

print("\nCluster Counts\n")
print(cluster_counts)

# Step 10: Identify Users in Each Cluster


for i in range(4):
    print(f"\nUsers in Cluster {i}")
    print(df[df["cluster"]==i]["user_id"].head())


# Step 11: PCA Visualization


pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df["pca1"] = X_pca[:,0]
df["pca2"] = X_pca[:,1]

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="pca1",
    y="pca2",
    hue="cluster",
    palette="Set2"
)

plt.title("User Segmentation Using PCA")

plt.show()


# Step 12: Export Results


df.to_csv("user_segmentation_results.csv", index=False)