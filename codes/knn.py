# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the dataset from a CSV file
file_path = "datasets\Homes for Sale and Real Estate.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Drop unnecessary columns for KNN
knn_data = df[['Beds', 'Bath', 'Sq.Ft', 'Place', 'Description']]

# Convert categorical variables into numerical format for KNN
label_encoder = LabelEncoder()
knn_data['Place'] = label_encoder.fit_transform(knn_data['Place'])
knn_data['Description'] = label_encoder.fit_transform(knn_data['Description'])

# Separate features and target variable
X = knn_data[['Beds', 'Bath', 'Sq.Ft', 'Place', 'Description']]
y_price = df['Price']
y_type = df['Description']

# Split the dataset into training and testing sets
X_train, X_test, y_price_train, y_price_test, y_type_train, y_type_test = train_test_split(
    X, y_price, y_type, test_size=0.2, random_state=42
)

# KNN for predicting housing prices
knn_price = KNeighborsRegressor(n_neighbors=5)
knn_price.fit(X_train, y_price_train)
y_price_pred = knn_price.predict(X_test)

# Evaluate the KNN model for housing prices
mse_price = mean_squared_error(y_price_test, y_price_pred)
print("\nKNN Model for Housing Prices:")
print("Mean Squared Error:", mse_price)

# KNN for predicting property types
knn_type = KNeighborsClassifier(n_neighbors=5)
knn_type.fit(X_train, y_type_train)
y_type_pred = knn_type.predict(X_test)

# Evaluate the KNN model for property types
accuracy_type = accuracy_score(y_type_test, y_type_pred)
print("\nKNN Model for Property Types:")
print("Accuracy Score:", accuracy_type)
