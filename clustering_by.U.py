import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Data baru
data = {
    'User_ID': ['U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08', 'U09', 'U10', 'U11', 'U12'],
    'Frekuensi_Topup': [1, 2, 1, 2, 3, 4, 3, 4, 5, 5, 4, 5],
    'Total_Data_GB': [5, 8, 7, 10, 18, 23, 18, 22, 35, 40, 38, 42],
    'Pengguna_per_Hari_GB': [1.5, 1.2, 0.7, 1.7, 2.5, 3.8, 3.3, 2.8, 5, 5.5, 6, 6.2]
}
df = pd.DataFrame(data)

# Normalisasi
scaler = StandardScaler()
X = scaler.fit_transform(df[['Frekuensi_Topup', 'Total_Data_GB', 'Pengguna_per_Hari_GB']])

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# Ambil centroid dan transform balik ke skala asli
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Warna dan label cluster
colors = ['gold', 'purple', 'teal']
labels = ['Light User', 'Regular User', 'Heavy User']
cluster_map = {i: labels[i] for i in range(3)}

# Plot
plt.figure(figsize=(10, 7))
for cluster_id in range(3):
    cluster_data = df[df['Cluster'] == cluster_id]
    plt.scatter(cluster_data['Total_Data_GB'], cluster_data['Pengguna_per_Hari_GB'],
                label=f"{labels[cluster_id]}", color=colors[cluster_id], s=100)
    for i, row in cluster_data.iterrows():
        plt.text(row['Total_Data_GB']+0.3, row['Pengguna_per_Hari_GB']+0.1, row['User_ID'])

# Plot centroids
plt.scatter(centroids[:, 1], centroids[:, 2], marker='X', s=200, c='black', label='Centroid')

plt.xlabel('Total Data (GB)')
plt.ylabel('Pengguna per Hari (GB)')
plt.title('K-Means Clustering Pengguna')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
