from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

def scale_data(df, attributes):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[attributes])
    return X_scaled

def find_optimal_k(X_scaled):
    inertia = []
    K = range(1, 15)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    return inertia

def cluster_data(X_scaled, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters
