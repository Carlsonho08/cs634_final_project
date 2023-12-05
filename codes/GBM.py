# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = "datasets\Homes for Sale and Real Estate.csv"
df = pd.read_csv(file_path)

# Select features for regression
features = df[['Beds', 'Bath', 'Sq.Ft']]

# Target variable
target = df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Gradient Boosting Regression
gb_regressor = GradientBoostingRegressor(random_state=42)
gb_regressor.fit(X_train, y_train)

# Predict on the test set
y_gb_pred = gb_regressor.predict(X_test)

# Evaluate performance
mse_gb = mean_squared_error(y_test, y_gb_pred)
r2_gb = r2_score(y_test, y_gb_pred)

# Display results
print("Gradient Boosting Regression:")
print("Mean Squared Error:", mse_gb)
print("R-squared:", r2_gb)

# Visualize predictions vs. actual values
plt.scatter(y_test, y_gb_pred, alpha=0.5)
plt.title('Gradient Boosting Regression: Predictions vs. Actual Values')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()
