import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# ---------------------------------------------------------
# 1. Load the Dataset
# ---------------------------------------------------------
# We'll load the dataset that contains Location, Year, Price, and Demand.
print("Loading real estate data...")
df = pd.read_csv("real_estate_data.csv")

# ---------------------------------------------------------
# 2. Analyze Current Patterns
# ---------------------------------------------------------
# Let's see the total growth in price from 2018 to 2023 for each location
price_2018 = df[df['Year'] == 2018].set_index('Location')['Price']
price_2023 = df[df['Year'] == 2023].set_index('Location')['Price']

# Calculate percentage growth
growth = ((price_2023 - price_2018) / price_2018) * 100
growth_df = growth.reset_index()
growth_df.columns = ['Location', 'Price Growth (%)']
growth_df = growth_df.sort_values(by='Price Growth (%)', ascending=False)

print("\n--- Price Growth (2018 - 2023) ---")
print(growth_df.to_string(index=False))

# Identify High Growth Areas (Growth > 50%)
high_growth_areas = growth_df[growth_df['Price Growth (%)'] > 50]['Location'].tolist()
print(f"\nHigh Growth Potential Areas Identified: {', '.join(high_growth_areas)}")

# ---------------------------------------------------------
# 3. Visualizations
# ---------------------------------------------------------
print("\nGenerating visualizations...")
sns.set_theme(style="whitegrid")

# Visualization A: Line Graph showing Price Trends over Years
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Year', y='Price', hue='Location', marker='o')
plt.title('Real Estate Price Trends (2018-2023)')
plt.ylabel('Price ($)')
plt.xlabel('Year')
plt.tight_layout()
plt.savefig('price_trends_linegraph.png')
plt.close()

# Visualization B: Bar Chart for Overall Price Growth
plt.figure(figsize=(8, 5))
sns.barplot(data=growth_df, x='Location', y='Price Growth (%)', palette='viridis')
plt.title('Overall Price Growth by Location (2018-2023)')
plt.ylabel('Growth (%)')
plt.tight_layout()
plt.savefig('price_growth_barchart.png')
plt.close()

# Visualization C: Heatmap of Correlation
# We will pivot the data to see the correlation of demand across different locations over years
heatmap_data = df.pivot(index="Location", columns="Year", values="Demand")
plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt="g")
plt.title('Heatmap: Real Estate Demand over Time')
plt.tight_layout()
plt.savefig('demand_heatmap.png')
plt.close()

# ---------------------------------------------------------
# 4. Predict Future Growth Trend (Simple Machine Learning)
# ---------------------------------------------------------
print("\nPredicting future prices for 2025 using Linear Regression...")

future_year = 2025
predictions = []

for loc in df['Location'].unique():
    # Filter data for the specific location
    loc_data = df[df['Location'] == loc]
    
    # Prepare X (features: Year) and y (target: Price)
    X = loc_data[['Year']]
    y = loc_data['Price']
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict the price for the future year (2025)
    # create a DataFrame with valid feature names
    X_future = pd.DataFrame({'Year': [future_year]})
    predicted_price = model.predict(X_future)[0]
    
    predictions.append({'Location': loc, 'Predicted Price (2025)': predicted_price})

predictions_df = pd.DataFrame(predictions)
print("\n--- Price Predictions for 2025 ---")
print(predictions_df.to_string(index=False))
print("\nAnalysis complete! Visualizations saved as PNG files.")
