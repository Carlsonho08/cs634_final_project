# Import necessary libraries
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load the dataset
file_path = "datasets\Homes for Sale and Real Estate.csv"
df = pd.read_csv(file_path)

# Select features for outlier detection
features = df[['Beds', 'Bath', 'Sq.Ft', 'Price']]

# Isolation Forest model
isolation_forest = IsolationForest(contamination=0.05, random_state=42)  # Adjust contamination based on your dataset
outlier_labels = isolation_forest.fit_predict(features)

# Add outlier labels to the original DataFrame
df['Outlier'] = outlier_labels

# Display the properties and their outlier status
print("Properties and Outlier Status:")
print(df[['Address', 'Beds', 'Bath', 'Sq.Ft', 'Price', 'Outlier']])

# Visualize outliers
plt.figure(figsize=(10, 6))
plt.scatter(df['Price'], df['Sq.Ft'], c=df['Outlier'], cmap='viridis', alpha=0.5)
plt.xlabel('Price')
plt.ylabel('Square Footage')
plt.title('Outlier Detection: Price vs. Square Footage')
plt.colorbar(label='Outlier Status')
plt.show()
