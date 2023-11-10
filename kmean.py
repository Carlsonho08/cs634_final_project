# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "./datasets/Homes for Sale and Real Estate.csv"
data = {
    'Address': ['3704 42 St SW', '30 Mahogany Mews SE #415', '273 Auburn Shores Way SE', '235 15 Ave SW #404',
                '24 Hemlock Crescent SW #2308', '591 Aboyne Crescent NE', '3406 64 St NE', '10551 Shillington Crescent SW',
                '928 3 Ave NW #1', '61 Royal Elm Green NW', '79 Evansfield Rd NW', '100 Royal Elm Green NW',
                '3234 New Brighton Gardens SE'],
    'Price': [979999, 439900, 950000, 280000, 649000, 434900, 419900, 499900, 269900, 627900, 910000, 939750, 399900],
    'Beds': [4, 2, 4, 2, 2, 6, 3, 4, 2, 2, 4, 4, 3],
    'Bath': [3.5, 2, 2.5, 2, 2, 2, 2.5, 2, 1, 2.5, 3.5, 3.5, 2.5],
    'Sq.Ft': [1813, 1029, 2545, 898, 1482, 1059, 1218, 1133, 756, 1303, 2500, 1747, 1125]
}

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
