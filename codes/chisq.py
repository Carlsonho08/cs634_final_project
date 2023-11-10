# Import necessary libraries
import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset from a CSV file
file_path = "../datasets/Homes for Sale and Real Estate.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Create a contingency table (cross-tabulation) for the Chi-square test
contingency_table = pd.crosstab(df['Place'], df['Description'])

# Print the contingency table
print("Contingency Table:")
print(contingency_table)

# Perform the Chi-square test
chi2, p, _, _ = chi2_contingency(contingency_table)

# Print the results
print("\nChi-square Statistic:", chi2)
print("P-value:", p)

# Interpret the results
alpha = 0.05
print("\nSignificance level (alpha):", alpha)
print("Conclusion:")
if p < alpha:
    print("Reject the null hypothesis. There is a significant association between property location and property type.")
else:
    print("Fail to reject the null hypothesis. There is no significant association between property location and property type.")
