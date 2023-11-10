# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset from a CSV file
file_path = "datasets\Homes for Sale and Real Estate.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Extract property descriptions
descriptions = df['Description']

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Fit and transform the property descriptions
tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Display the TF-IDF matrix
print("TF-IDF Matrix:")
print(tfidf_df)

# Example: Display the top keywords for each property description
for i, description in enumerate(descriptions):
    print(f"\nTop keywords for Property {i + 1}: {description}")
    keywords_indices = tfidf_df.iloc[i].sort_values(ascending=False).head(5).index
    print(tfidf_vectorizer.get_feature_names_out()[keywords_indices])
