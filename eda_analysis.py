import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv('eda_india_housing_prices.csv')

# 1. Distribution of property prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Price_in_Lakhs'], kde=True, color='blue')
plt.title('Distribution of Property Prices (in Lakhs)')
plt.show()

# 2. Distribution of property sizes
plt.figure(figsize=(10, 6))
sns.histplot(df['Size_in_SqFt'], kde=True, color='green')
plt.title('Distribution of Property Sizes (SqFt)')
plt.show()

# 3. Price per sq ft variation by property type
plt.figure(figsize=(12, 6))
sns.boxplot(x='Property_Type', y='Price_per_SqFt', data=df)
plt.title('Price per SqFt by Property Type')
plt.xticks(rotation=45)
plt.show()

# 4. Relationship between property size and price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Size_in_SqFt', y='Price_in_Lakhs', data=df, alpha=0.5)
plt.title('Property Size vs. Total Price')
plt.show()

# 5. Outliers in price per sq ft or property size
# (Using boxplots is the best way to visually identify these)
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['Price_per_SqFt'])
plt.title('Identifying Outliers in Price per SqFt')
plt.show()

# 6. What is the average price per sq ft by state?
plt.figure(figsize=(12, 6))
df.groupby('State')['Price_per_SqFt'].mean().sort_values().plot(kind='bar', color='skyblue')
plt.title('Average Price per SqFt by State')
plt.ylabel('Price per SqFt')
plt.show()

# 7. What is the average property price by city?
plt.figure(figsize=(12, 6))
# Plotting top 10 cities for better visibility
df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10).plot(kind='bar', color='coral')
plt.title('Top 10 Cities by Average Property Price')
plt.ylabel('Price in Lakhs')
plt.show()

# 8. What is the median age of properties by locality?
plt.figure(figsize=(12, 6))
# Analyzing the top 10 most frequent localities in the dataset
top_localities = df['Locality'].value_counts().head(10).index
sns.barplot(x='Locality', y='Age_of_Property', data=df[df['Locality'].isin(top_localities)], estimator='median')
plt.title('Median Property Age by Top 10 Localities')
plt.xticks(rotation=45)
plt.show()

# 9. How is BHK distributed across cities?
plt.figure(figsize=(14, 7))
# Focusing on the top 5 cities to avoid a cluttered chart
top_5_cities = df['City'].value_counts().head(5).index
sns.countplot(x='City', hue='BHK', data=df[df['City'].isin(top_5_cities)])
plt.title('BHK Distribution in Top 5 Cities')
plt.show()

# 10. What are the price trends for the top 5 most expensive localities?
plt.figure(figsize=(12, 6))
top_5_expensive_localities = df.groupby('Locality')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(5).index
sns.boxplot(x='Locality', y='Price_in_Lakhs', data=df[df['Locality'].isin(top_5_expensive_localities)])
plt.title('Price Distribution in Top 5 Most Expensive Localities')
plt.xticks(rotation=45)
plt.show()

# 11. How are numeric features correlated with each other?
plt.figure(figsize=(12, 10))
# Calculate correlation on numeric columns only
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='RdYlGn', fmt='.2f')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

# 12. How do nearby schools relate to price per sq ft?
plt.figure(figsize=(10, 6))
sns.regplot(x='Nearby_Schools', y='Price_per_SqFt', data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Impact of Nearby Schools on Price per SqFt')
plt.show()

# 13. How do nearby hospitals relate to price per sq ft?
plt.figure(figsize=(10, 6))
sns.boxplot(x='Nearby_Hospitals', y='Price_per_SqFt', data=df)
plt.title('Price per SqFt vs. Nearby Hospitals Count')
plt.show()

# 14. How does price vary by furnished status?
plt.figure(figsize=(10, 6))
sns.violinplot(x='Furnished_Status', y='Price_in_Lakhs', data=df)
plt.title('Price Distribution by Furnished Status')
plt.show()

# 15. How does price per sq ft vary by property facing direction?
plt.figure(figsize=(12, 6))
# Fixed: Assigned 'Facing' to both x and hue
sns.barplot(x='Facing', y='Price_per_SqFt', data=df, hue='Facing', palette='viridis', legend=False)
plt.title('Average Price per SqFt by Facing Direction')
plt.show()

# 16. How many properties belong to each owner type?
plt.figure(figsize=(10, 6))
df['Owner_Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Property Distribution by Owner Type')
plt.ylabel('')
plt.show()

# 17. How many properties are available under each availability status?
plt.figure(figsize=(10, 6))
# Fixed: Assigned 'Availability_Status' to both x and hue
sns.countplot(x='Availability_Status', data=df, hue='Availability_Status', palette='Set2', legend=False)
plt.title('Property Count by Availability Status')
plt.show()

# 18. Does parking space affect property price?
plt.figure(figsize=(10, 6))
sns.barplot(x='Parking_Space', y='Price_in_Lakhs', data=df, estimator='mean')
plt.title('Average Price vs. Number of Parking Spaces')
plt.show()

# 19. How do amenities affect price per sq ft?
plt.figure(figsize=(10, 6))

# Clean the column: fill missing with 'Unknown' and ensure it is a string
df['Amenities'] = df['Amenities'].fillna('Unknown').astype(str)

# Only include categories that actually exist in your CSV
existing_categories = [cat for cat in ['Low', 'Medium', 'High'] if cat in df['Amenities'].unique()]

if existing_categories:
    sns.boxplot(x='Amenities', y='Price_per_SqFt', data=df, order=existing_categories)
    plt.title('Impact of Amenities Level on Price per SqFt')
    plt.show()
else:
    # Fallback if the labels 'Low/Medium/High' aren't in your specific dataset
    sns.boxplot(x='Amenities', y='Price_per_SqFt', data=df)
    plt.title('Impact of Amenities on Price per SqFt')
    plt.show()

# 20. How does public transport accessibility relate to investment potential?
plt.figure(figsize=(10, 6))
sns.countplot(x='Public_Transport_Accessibility', hue='Good_Investment', data=df)
plt.title('Transport Accessibility vs. Good Investment Classification')
plt.show()