import pandas as pd
import numpy as np
from data_preprocessing import load_data, preprocess_data
from analysis import get_top_attributes, weight_top_attributes, rename_cluster, get_top_players_per_cluster
from clustering import scale_data, find_optimal_k, cluster_data
from visualization import plot_elbow_curve, plot_pca, plot_tsne, plot_radar_chart, plot_heatmap, plot_parallel_coordinates

# Adjust path relative to the location of run_clustering.py
file_path = '../data/fifa_23_players.csv'

# Step 1: Load and preprocess the data
df = load_data(file_path)
df, attributes = preprocess_data(df)

# Step 2: Determine top attributes and apply weighting
df['top_10'] = df.apply(lambda row: get_top_attributes(row, attributes).index.tolist(), axis=1)
df[attributes] = df.apply(lambda row: weight_top_attributes(row[attributes], attributes, row['top_10']), axis=1)

# Step 3: Scale the data
X_scaled = scale_data(df, attributes)

# Step 4: Determine the optimal number of clusters using the Elbow method
inertia = find_optimal_k(X_scaled)
plot_elbow_curve(inertia)
optimal_k = max(6, np.argmin(np.diff(inertia)) + 2)  # Assuming the desired minimum is 6 clusters

# Step 5: Cluster the data
df['cluster'] = cluster_data(X_scaled, optimal_k)

# Step 6: Rename clusters based on the most common positions
df['cluster_name'] = df['cluster'].apply(lambda x: rename_cluster(x, df))

# Step 7: Visualize the clusters
plot_pca(X_scaled, df, 'cluster_name')
plot_tsne(X_scaled, df, 'cluster_name')
plot_radar_chart(df, attributes)
plot_heatmap(df, attributes)
plot_parallel_coordinates(df, attributes)

# Step 8: Additional Analysis - Top 10 Players in Each Cluster
top_players_per_cluster = get_top_players_per_cluster(df, top_n=10)
for cluster, players_df in top_players_per_cluster.items():
    print(f"Top Players in {rename_cluster(cluster, df)}:")
    print(players_df)
    print("\n")
