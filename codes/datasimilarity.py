# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Add LabelEncoder here

# Load the dataset from a CSV file
file_path = "datasets\Homes for Sale and Real Estate.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Drop unnecessary columns for similarity analysis
similarity_data = df[['Beds', 'Bath', 'Sq.Ft', 'Place', 'Description']]

# Convert categorical variables into numerical format for similarity analysis
label_encoder = LabelEncoder()
similarity_data['Place'] = label_encoder.fit_transform(similarity_data['Place'])
similarity_data['Description'] = label_encoder.fit_transform(similarity_data['Description'])

# Standardize the features for similarity analysis
scaler = StandardScaler()
similarity_data_scaled = scaler.fit_transform(similarity_data[['Beds', 'Bath', 'Sq.Ft', 'Place', 'Description']])

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(similarity_data_scaled, similarity_data_scaled)

# Display the cosine similarity matrix
print("Cosine Similarity Matrix:")
print(cosine_sim_matrix)

# Example: Find similar properties to a given property (e.g., first property in the dataset)
property_index = 0
similar_properties = cosine_sim_matrix[property_index]

# Display the most similar properties
print("\nMost Similar Properties to Property at Index", property_index)
similar_properties_indices = sorted(range(len(similar_properties)), key=lambda i: similar_properties[i], reverse=True)[1:6]
for index in similar_properties_indices:
    print(f"Property at Index {index}: Cosine Similarity = {similar_properties[index]}")
    print(df.loc[index, ['Address', 'Price', 'Beds', 'Bath', 'Sq.Ft', 'Place', 'Description']])
    print("\n")
