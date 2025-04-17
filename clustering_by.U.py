import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Buat Data Simulasi
# -------------------------------
data = {
    'User_ID': ['U001', 'U002', 'U003', 'U004', 'U005', 'U006', 'U007', 'U008', 'U009', 'U010'],
    'Frekuensi_Topup': [5, 1, 3, 4, 2, 5, 2, 1, 3, 4],
    'Total_Data_GB': [20, 5, 30, 25, 10, 22, 12, 6, 18, 28],
    'Penggunaan_per_Hari_GB': [0.7, 0.2, 1.2, 1.0, 0.3, 0.8, 0.4, 0.1, 0.6, 1.1],
    'Durasi_Aktif_bulan': [12, 3, 9, 10, 5, 14, 6, 2, 8, 11]
}

df = pd.DataFrame(data)
df.set_index('User_ID', inplace=True)

# -------------------------------
# 2. Preprocessing
# -------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# -------------------------------
# 3. Elbow Method (untuk cari jumlah cluster optimal)
# -------------------------------
inertia = []
for k in range(1, 10):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_data)
    inertia.append(model.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.title('Elbow Method - Cari Jumlah Cluster Optimal')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia')
plt.grid()
plt.show()

# -------------------------------
# 4. Clustering dengan K-Means
# -------------------------------
k = 3  # Misal hasil elbow nunjuk 3
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# -------------------------------
# 5. Visualisasi Cluster
# -------------------------------
sns.scatterplot(x='Total_Data_GB', y='Penggunaan_per_Hari_GB', hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Visualisasi Clustering Pengguna by.U')
plt.xlabel('Total Data Dibeli (GB)')
plt.ylabel('Penggunaan per Hari (GB)')
plt.grid()
plt.show()

# -------------------------------
# 6. Lihat Hasil
# -------------------------------
print(df.sort_values('Cluster'))
