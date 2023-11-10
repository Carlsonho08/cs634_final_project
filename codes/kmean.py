# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "datasets\Homes for Sale and Real Estate.csv"


df = pd.read_csv(file_path)

# Select features for clustering
features = df[['Price', 'Beds', 'Bath', 'Sq.Ft']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Determine the number of clusters (you can adjust this based on your requirements)
num_clusters = 3

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)

# Display the clustered properties
print("Clustered Properties:")
print(df[['Address', 'Price', 'Beds', 'Bath', 'Sq.Ft', 'Cluster']])

# Visualize the clusters (for 2D visualization)
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('Standardized Price')
plt.ylabel('Standardized Beds')
plt.title('K-means Clustering of Properties')
plt.show()
