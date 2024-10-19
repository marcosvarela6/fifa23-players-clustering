# FIFA 23 Player Clustering Project

This project clusters FIFA 23 players based on their attributes to find similar groups of players. The clustering is performed using only specific player attributes, excluding general attributes such as overall rating, position, and overall defensive or attacking attributes. This approach focuses on a detailed analysis of attributes like crossing, finishing, dribbling, and more, to group players based on their nuanced skill sets.

## Project Structure

- **data/**: Contains the input data file `fifa_23_players.csv`.
- **src/**: Contains all the Python scripts used for data processing, clustering, and visualization.
  - `data_preprocessing.py`: Script for data loading and preprocessing.
  - `analysis.py`: Functions for analyzing player attributes and clusters.
  - `clustering.py`: Functions for scaling, finding the optimal number of clusters, and clustering the data.
  - `visualization.py`: Functions for visualizing the clustering results.
  - `run_clustering.py`: Main script to run the entire clustering process.
- **requirements.txt**: List of dependencies required to run the project.
- **README.md**: Overview and instructions for the project.

## How to Run

1. Install the required packages using:
   ```bash
   pip install -r requirements.txt

2. Run the script:
   ```bash
   python src/run_clustering.py


