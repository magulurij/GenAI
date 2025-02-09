import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the cleaned dataset
data = pd.read_csv(r'C:\Users\srava\Downloads\archive (13)\Metro_Interstate_Traffic_Volume.csv')

# Step 1: Data Cleaning
# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Check for duplicates
print("\nDuplicate rows in the dataset:")
print(data.duplicated().sum())

# Step 2: Outlier Detection with Box Plots for Numerical Columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Create box plots for each numerical column
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=data[column])
    plt.title(f'Box Plot for {column}')
    plt.show()

# Step 3: Exploratory Data Analysis (EDA)
# Histograms for each numerical column
data[numerical_columns].hist(bins=15, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Features', fontsize=16)
plt.show()

# Pair plot to observe relationships between features
sns.pairplot(data)
plt.suptitle('Pair Plot of Features', fontsize=16)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Step 4: Splitting Data into Training and Testing Sets
# Assuming 'traffic_volume' is the target variable (replace with actual column name if different)
X = data.drop('traffic_volume', axis=1)  # Features
y = data['traffic_volume']  # Target variable

# Convert any categorical columns if present (optional)
X = pd.get_dummies(X, drop_first=True)

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Training a Linear Regression Model for Traffic Volume Prediction
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predicting and Testing the Model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate model performance
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("\n--- Traffic Volume Prediction Results ---")
print(f"Training MSE: {mse_train:.2f}")
print(f"Training R^2 Score: {r2_train:.2f}")
print(f"Testing MSE: {mse_test:.2f}")
print(f"Testing R^2 Score: {r2_test:.2f}")

# Step 7: Visualization of Predictions vs Actual Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='blue', label='Predicted Traffic Volume')
plt.plot(y_test, y_test, color='red', label='Actual Traffic Volume', linewidth=2)
plt.title('Predicted vs Actual Traffic Volume (Test Set)')
plt.xlabel('Actual Traffic Volume')
plt.ylabel('Predicted Traffic Volume')
plt.legend()
plt.show()
