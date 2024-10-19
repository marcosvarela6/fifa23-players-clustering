import pandas as pd

def get_top_attributes(player_row, attributes, top_n=10):
    player_attributes = player_row[attributes].astype(float)
    top_attributes = player_attributes.nlargest(top_n)
    return top_attributes

def weight_top_attributes(row, attributes, top_10_attributes, weight_factor=10):
    weighted_row = row.copy()
    for attr in top_10_attributes:
        weighted_row[attr] *= weight_factor
    return weighted_row

def rename_cluster(cluster, df):
    cluster_df = df[df['cluster'] == cluster]
    position_counts = cluster_df['best_position'].value_counts(normalize=True) * 100
    position_counts = position_counts[position_counts >= 14]

    cumulative_percentage = 0
    included_positions = []

    for position, percentage in position_counts.items():
        if cumulative_percentage >= 82:
            break
        cumulative_percentage += percentage
        included_positions.append(position)

    if not included_positions:
        return f'Cluster {cluster}'
    elif len(included_positions) == 1:
        return f'Cluster {included_positions[0]}'
    elif len(included_positions) == 2:
        return f'Cluster {included_positions[0]}/{included_positions[1]}'
    else:
        return f'Cluster {included_positions[0]}/{included_positions[1]}/{included_positions[2]}'

def get_top_players_per_cluster(df, top_n=10):
    top_players_dict = {}

    for cluster in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster]
        top_players = cluster_df.sort_values(by='overall', ascending=False).head(top_n)
        top_players_dict[cluster] = top_players[['know_as', 'full_name', 'overall', 'best_position']]

    return top_players_dict
