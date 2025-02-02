# Internship-task-2
The aim of this data analytics project is to perform customer segmentation analysis for an e- commerce company. By analyzing customer behavior and purchase patterns, the goal is to group customers into distinct segments. 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('/content/ifood_df.csv')

# 1. Data Overview
print("Data Overview:")
print(df.head())  # Show first 5 rows
print("\nData Shape:")
print(df.shape)  # Show number of rows and columns
print("\nData Info:")
print(df.info())  # Show data types and missing values
print("\nData Summary:")
print(df.describe())  # Show summary statistics

# 2. Missing Data Handling
print("\nMissing Data:")
missing_data = df.isnull().sum()
print(missing_data)  # Display missing values in each column

# Visualize missing data
plt.figure(figsize=(10, 7))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.tight_layout()  # Adjust the layout to avoid title overlap
plt.show()

# 3. Data Distribution and Summary Statistics
# Plot histograms for numerical columns
df.hist(bins=15, figsize=(15, 10))
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()  # Adjust the layout to avoid title overlap
plt.show()

# 4. Correlation Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix", fontsize=16)
plt.tight_layout()  # Adjust the layout to avoid title overlap
plt.show()

# 5. Feature Relationships (Scatter plots)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Income', y='MntTotal')
plt.title('Income vs. Total Spend (MntTotal)', fontsize=14)
plt.tight_layout()  # Adjust the layout to avoid title overlap
plt.show()

# Pair plot to see interaction between multiple features
sns.pairplot(df[['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntTotal']])
plt.suptitle("Pairplot of Selected Features", y=1.02, fontsize=16)
plt.tight_layout()  # Adjust the layout to avoid title overlap
plt.show()

# 6. Outlier Detection
plt.figure(figsize=(8, 6))
sns.boxplot(x='Income', data=df)
plt.title('Income Distribution with Outliers', fontsize=14)
plt.tight_layout()  # Adjust the layout to avoid title overlap
plt.show()

# 7. Class Distribution (Cluster Count)
# Assuming KMeans clustering has been applied and 'Cluster' column is in df
kmeans = KMeans(n_clusters=3)  # Example: 3 clusters
# Replace 'feature1' and 'feature2' with your actual feature names
# Example: Using 'Income' and 'MntTotal' for clustering
df['Cluster'] = kmeans.fit_predict(df[['Income', 'MntTotal']])

# Now df['Cluster'] exists, you can plot the distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', data=df)
plt.title("Cluster Distribution", fontsize=14)
plt.tight_layout()  # Adjust the layout to avoid title overlap
plt.show()

# 8. Feature Engineering (if necessary)
# Example: Log transformation for skewed 'Income' column
df['Log_Income'] = df['Income'].apply(lambda x: np.log(x + 1))
sns.histplot(df['Log_Income'], kde=True)
plt.title("Log-Transformed Income Distribution", fontsize=14)
plt.tight_layout()  # Adjust the layout to avoid title overlap
plt.show()
