# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
file_path = "datasets\Homes for Sale and Real Estate.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Create a simplified dataset for illustration
# You may need to adapt this to your actual dataset and analysis
simplified_df = df[['Place', 'Beds', 'Price']]

# Group by 'Place' and calculate average price
average_price_by_place = simplified_df.groupby('Place')['Price'].mean().reset_index()

# Plot the overall average price
plt.figure(figsize=(10, 5))
sns.barplot(x='Place', y='Price', data=average_price_by_place, color='lightblue')
plt.title('Overall Average Price by Placce')
plt.show()

# Apply Simpson's Paradox by introducing a confounding variable ('Beds')
plt.figure(figsize=(12, 6))
sns.barplot(x='Place', y='Price', hue='Beds', data=simplified_df, palette='viridis')
plt.title('Average Price by Place and Number of Bedrooms')
plt.show()
