# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules

# Load the dataset from a CSV file
file_path = "../datasets/Homes for Sale and Real Estate.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Drop unnecessary columns for market basket analysis
basket_data = df[['Beds', 'Bath', 'Place']]

# Convert categorical variables into a suitable format for analysis
basket_encoded = pd.get_dummies(basket_data)

# Apriori
apriori_result = apriori(basket_encoded, min_support=0.1, use_colnames=True)

# FP-growth
fp_growth_result = fpgrowth(basket_encoded, min_support=0.1, use_colnames=True)

# Display the results
print("Apriori Results:")
print(apriori_result)

print("\nFP-growth Results:")
print(fp_growth_result)

# Generate association rules for Apriori
apriori_rules = association_rules(apriori_result, metric="lift", min_threshold=1.0)
print("\nApriori Association Rules:")
print(apriori_rules)

# Generate association rules for FP-growth
fp_growth_rules = association_rules(fp_growth_result, metric="lift", min_threshold=1.0)
print("\nFP-growth Association Rules:")
print(fp_growth_rules)
