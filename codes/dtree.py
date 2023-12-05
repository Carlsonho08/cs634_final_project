# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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

# Decision Tree Regression
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Predict on the test set
y_dt_pred = dt_regressor.predict(X_test)

# Random Forest Regression
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict on the test set
y_rf_pred = rf_regressor.predict(X_test)

# Evaluate performance
mse_dt = mean_squared_error(y_test, y_dt_pred)
r2_dt = r2_score(y_test, y_dt_pred)

mse_rf = mean_squared_error(y_test, y_rf_pred)
r2_rf = r2_score(y_test, y_rf_pred)

# Display results
print("Decision Tree Regression:")
print("Mean Squared Error:", mse_dt)
print("R-squared:", r2_dt)

print("\nRandom Forest Regression:")
print("Mean Squared Error:", mse_rf)
print("R-squared:", r2_rf)

# Visualize predictions vs. actual values
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.scatter(y_test, y_dt_pred, alpha=0.5)
plt.title('Decision Tree Regression: Predictions vs. Actual Values')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

plt.subplot(2, 1, 2)
plt.scatter(y_test, y_rf_pred, alpha=0.5)
plt.title('Random Forest Regression: Predictions vs. Actual Values')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

plt.tight_layout()
plt.show()
