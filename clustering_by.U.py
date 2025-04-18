import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Data pengguna
data = {
    'User_ID': ['U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08', 'U09', 'U10', 'U11', 'U12'],
    'Frekuensi_Topup': [1, 2, 1, 2, 3, 4, 3, 4, 5, 5, 4, 5],
    'Total_Data_GB': [5, 8, 7, 10, 18, 23, 18, 22, 35, 40, 38, 42],
    'Pengguna_per_Hari_GB': [1.5, 1.2, 0.7, 1.7, 2.5, 3.8, 3.3, 2.8, 5.0, 5.5, 6.0, 6.2]
}

# Buat DataFrame
df = pd.DataFrame(data)

# Fitur yang digunakan untuk clustering
X = df[['Frekuensi_Topup', 'Total_Data_GB', 'Pengguna_per_Hari_GB']]

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering dengan KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Menampilkan hasil
print(df)

# Visualisasi klaster (2D menggunakan dua fitur utama)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='Total_Data_GB',
    y='Pengguna_per_Hari_GB',
    hue='Cluster',
    palette='Set2',
    s=100
)
plt.title('Hasil Clustering Pengguna by.U')
plt.xlabel('Total Data (GB)')
plt.ylabel('Penggunaan Harian (GB/hari)')
plt.grid(True)
plt.show()
