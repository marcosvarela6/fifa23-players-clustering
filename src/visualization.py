import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import pi
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pandas.plotting import parallel_coordinates

def plot_elbow_curve(inertia):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 15), inertia, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def plot_pca(X_scaled, df, cluster_col):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X_scaled)
    df['pca_one'] = pca_results[:, 0]
    df['pca_two'] = pca_results[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='pca_one', y='pca_two', hue=cluster_col, data=df, palette='tab10', s=100, alpha=0.7)
    plt.title('PCA: Player Clusters Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster', loc='upper right')
    plt.show()

def plot_tsne(X_scaled, df, cluster_col):
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(X_scaled)
    df['tsne_one'] = tsne_results[:, 0]
    df['tsne_two'] = tsne_results[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='tsne_one', y='tsne_two', hue=cluster_col, data=df, palette='tab10', s=100, alpha=0.7)
    plt.title('t-SNE: Player Clusters Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Cluster', loc='upper right')
    plt.show()

def plot_radar_chart(df, attributes):
    cluster_centers_df = df.groupby('cluster').mean()
    for i in range(len(cluster_centers_df)):
        values = cluster_centers_df.loc[i, attributes].values.flatten().tolist()
        values += values[:1]  # repeat the first value to close the circular graph

        angles = [n / float(len(attributes)) * 2 * pi for n in range(len(attributes))]
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], attributes, color='grey', size=8)
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color='blue', alpha=0.25)
        plt.title(f'Cluster {i} Profile')
        plt.show()

def plot_heatmap(df, attributes):
    cluster_centers_df = df.groupby('cluster').mean()
    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_centers_df[attributes], annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Cluster Centers Heatmap')
    plt.xlabel('Attributes')
    plt.ylabel('Clusters')
    plt.show()

def plot_parallel_coordinates(df, attributes):
    cluster_centers_df = df.groupby('cluster').mean().reset_index()
    plt.figure(figsize=(14, 8))
    parallel_coordinates(cluster_centers_df, class_column='cluster', cols=attributes, color=sns.color_palette('tab10'))
    plt.title('Parallel Coordinates Plot for Cluster Centers')
    plt.xticks(rotation=90)
    plt.show()
