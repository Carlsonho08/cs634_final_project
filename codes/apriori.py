# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the real estate dataset
file_path = "datasets\Homes for Sale and Real Estate.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Select relevant columns for analysis
property_data = df[['Beds', 'Bath', 'Place']]

# Convert categorical variables into numerical format for Apriori and FP-growth
property_data_encoded = pd.get_dummies(property_data, columns=['Beds', 'Bath', 'Place'])

# Apriori algorithm to find frequent itemsets
frequent_itemsets_apriori = apriori(property_data_encoded, min_support=0.1, use_colnames=True)

# FP-growth algorithm to find frequent itemsets
frequent_itemsets_fpgrowth = fpgrowth(property_data_encoded, min_support=0.1, use_colnames=True)

# Display the frequent itemsets
print("Frequent Itemsets using Apriori:")
print(frequent_itemsets_apriori)

print("\nFrequent Itemsets using FP-growth:")
print(frequent_itemsets_fpgrowth)

# Generate association rules using Apriori
rules_apriori = association_rules(frequent_itemsets_apriori, metric='lift', min_threshold=1.0)

# Generate association rules using FP-growth
rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric='lift', min_threshold=1.0)

# Display the generated association rules
print("\nAssociation Rules using Apriori:")
print(rules_apriori)

print("\nAssociation Rules using FP-growth:")
print(rules_fpgrowth)
