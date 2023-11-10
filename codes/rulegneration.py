# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelEncoder

# Load the dataset from a CSV file
file_path = "datasets\Homes for Sale and Real Estate.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Drop unnecessary columns for rule generation
rule_data = df[['Beds', 'Bath', 'Sq.Ft', 'Place', 'Description', 'Price']]

# Convert categorical variables into numerical format for rule generation
label_encoder = LabelEncoder()
rule_data['Place'] = label_encoder.fit_transform(rule_data['Place'])
rule_data['Description'] = label_encoder.fit_transform(rule_data['Description'])

# Bin the continuous variable (Price) to create categorical labels
bins = [0, 500000, 1000000, float('inf')]
labels = ['Low', 'Medium', 'High']
rule_data['Price_Category'] = pd.cut(rule_data['Price'], bins=bins, labels=labels, right=False)

# Drop the original Price column
rule_data = rule_data.drop('Price', axis=1)

# Convert the dataset to a one-hot encoded format
one_hot_encoded = pd.get_dummies(rule_data, columns=['Beds', 'Bath', 'Sq.Ft', 'Place', 'Description', 'Price_Category'])

# Apriori to find frequent itemsets
frequent_itemsets = apriori(one_hot_encoded, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

# Display the generated association rules
print("Generated Association Rules:")
print(rules)
