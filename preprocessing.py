import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('india_housing_prices.csv')
# Fill missing values
df['Parking_Space'] = df['Parking_Space'].fillna(0)
df['Nearby_Schools'] = df['Nearby_Schools'].fillna(df['Nearby_Schools'].median())
df['Nearby_Hospitals'] = df['Nearby_Hospitals'].fillna(df['Nearby_Hospitals'].median())
df['Furnished_Status'] = df['Furnished_Status'].fillna('Unfurnished')

# Remove duplicates
df.drop_duplicates(inplace=True)

# Basic Feature Calculations
df['Price_per_SqFt'] = df['Price_in_Lakhs'] / df['Size_in_SqFt']
df['Age_of_Property'] = 2026 - df['Year_Built']

# Regression Target: Estimated Price after 5 years (8% annual growth)
df['Future_Price_5Y'] = df['Price_in_Lakhs'] * (1.08 ** 5)

# Classification Target: Define "Good Investment" (1 = Yes, 0 = No)
# Logic: Price_per_SqFt is below average AND Accessibility is High
mean_price_sqft = df['Price_per_SqFt'].mean()

df['Good_Investment'] = np.where(
    (df['Price_per_SqFt'] < mean_price_sqft) & 
    (df['Public_Transport_Accessibility'] == 'High') &
    (df['Amenities'].str.contains('Gym|Pool', na=False)), 
    1, 0
)


# Convert categorical features to dummy variables
df_processed = pd.get_dummies(df, columns=['City', 'Property_Type', 'Furnished_Status'], drop_first=True)

# --- Modify this part in preprocessing.py ---

# 1. Save this version for EDA (Keep text labels)
df.to_csv('eda_india_housing_prices.csv', index=False) 

# 2. THEN do the encoding for Machine Learning later
df_processed = pd.get_dummies(df, columns=['City', 'Property_Type', 'Furnished_Status'], drop_first=True)
df_processed.to_csv('cleaned_india_housing_prices.csv', index=False)

print("Step 1 Complete: Both EDA and ML files saved.")
