# Import necessary libraries
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

# Load the dataset from a CSV file
file_path = "datasets\Homes for Sale and Real Estate.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Create a SQLite database and engine
db_path = "real_estate_database.db"  # Replace with the desired path for the database file
engine = create_engine(f"sqlite:///{db_path}")

# Save the DataFrame to the database
df.to_sql('real_estate', engine, index=False, if_exists='replace')

# OLAP Analysis Example: Average Price by Location and Property Type
query = """
    SELECT Place, Description, AVG(Price) as AvgPrice
    FROM real_estate
    GROUP BY Place, Description
"""

# Execute the query and fetch results
result_df = pd.read_sql(query, engine)

# Display the OLAP results
print("OLAP Analysis - Average Price by Location and Property Type:")
print(result_df)
