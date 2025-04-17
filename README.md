# Clustering by.U

This project implements a K-Means clustering algorithm to categorize users based on their mobile usage behavior for the "by.U" service. The dataset simulates information such as top-up frequency, total data usage, daily data consumption, and the duration of active months for each user. The goal is to identify patterns within the user data, helping to segment users into distinct clusters.

## Key Features
- **Data Preprocessing**: StandardScaler for feature scaling.
- **Elbow Method**: Used to determine the optimal number of clusters.
- **K-Means Clustering**: Applied to segment users into meaningful clusters.
- **Visualization**: Visualizing the clustering results using Matplotlib and Seaborn.

## Requirements
To run this project, you need to install the following Python libraries:

- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install these libraries using pip:

```bash
pip install pandas scikit-learn matplotlib seaborn
