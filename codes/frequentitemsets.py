# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelEncoder

# Load the dataset from a CSV file
file_path = "datasets\Homes for Sale and Real Estate.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Drop unnecessary columns for frequent itemset mining
itemset_data = df[['Beds', 'Bath', 'Sq.Ft', 'Place', 'Description']]

# Convert categorical variables into numerical format for frequent itemset mining
label_encoder = LabelEncoder()
itemset_data['Place'] = label_encoder.fit_transform(itemset_data['Place'])
itemset_data['Description'] = label_encoder.fit_transform(itemset_data['Description'])

# Convert the dataset to a one-hot encoded format
one_hot_encoded = pd.get_dummies(itemset_data, columns=['Beds', 'Bath', 'Sq.Ft', 'Place', 'Description'])

# Apriori to find frequent itemsets
frequent_itemsets = apriori(one_hot_encoded, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

# Display the generated frequent itemsets
print("Generated Frequent Itemsets:")
print(frequent_itemsets)

# Compact representation: Keep only relevant information
compact_representation = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# Display the compact representation
print("\nCompact Representation of Frequent Itemsets:")
print(compact_representation)
