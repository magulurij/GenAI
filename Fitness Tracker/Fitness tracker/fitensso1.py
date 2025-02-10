# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the Dataset
file_path = r'C:\Users\srava\Downloads\dailyActivity_merged.csv'
df = pd.read_csv(file_path)

# Initial Data Overview
print("Dataset Info:")
print(df.info())
print("\nDataset Head:")
print(df.head())

# Step 1: Data Cleaning
# Check for missing values
print("\nMissing Values in Dataset:")
print(df.isnull().sum())

# Drop date columns or non-numeric columns that are not needed for modeling
# Assuming columns like 'ActivityDate' might be present and not required for correlation or model training
df = df.select_dtypes(include=[np.number])  # Keep only numeric columns

# Fill any remaining missing values with the median
df.fillna(df.median(numeric_only=True), inplace=True)

# Step 2: Identifying Outliers
# Box plot to detect outliers in numerical features
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('Outliers in Numerical Columns')
plt.show()

# Step 3: Exploratory Data Analysis (EDA)
# Distribution of Variables
df.hist(figsize=(12, 10), bins=30)
plt.suptitle('Distribution of Features')
plt.show()

# Correlation Heatmap (with numeric columns only)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 4: Splitting the Dataset into Features and Target
# Assuming 'Calories' is the target column.
X = df.drop('Calories', axis=1)  # Features
y = df['Calories']               # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Training the Model
# Using Random Forest Regressor for calorie prediction
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 7: Making Predictions
y_pred = model.predict(X_test_scaled)

# Step 8: Evaluating the Model
# Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")

# Scatter plot of predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Actual vs Predicted Calories Burned")
plt.show()

# Step 9: Visualizing Important Features
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Calculating Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²) Score: {r2}")










