import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the cleaned dataset
data = pd.read_csv(r'C:\Users\srava\Downloads\Automobile.csv')

# Print column names to confirm correct fare column name
print("Columns in the dataset:")
print(data.columns)

# Step 1: Define Features and Target
X = data.drop('fare_amount', axis=1)  # Features
y = data['fare_amount']  # Target variable (fare price)

# Step 2: Data Cleaning Checks
print("\nMissing values in the dataset:")
print(data.isnull().sum())
print("\nDuplicate rows in the dataset:")
print(data.duplicated().sum())

# Step 3: Outlier Detection with Box Plots
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=data[column])
    plt.title(f'Box Plot for {column}')
    plt.show()

# Step 4: EDA - Histograms and Correlation Heatmap
data[numerical_columns].hist(bins=15, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Features', fontsize=16)
plt.show()

sns.pairplot(data)
plt.suptitle('Pair Plot of Features', fontsize=16)
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(data[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Step 5: Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Linear Regression Model for Fare Prediction
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predictions and Evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# Output model performance for Fare Prediction
print("\nFare Prediction - Training MSE:", mse_train)
print("Fare Prediction - Training R² Score:", r2_train)
print("Fare Prediction - Testing MSE:", mse_test)
print("Fare Prediction - Testing R² Score:", r2_test)

# Step 8: Visualization of Predictions vs Actual Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='blue', label='Predicted Fare')
plt.plot(y_test, y_test, color='red', label='Actual Fare', linewidth=2)
plt.title('Predicted vs Actual Fare Price (Test Set)')
plt.xlabel('Actual Fare Price')
plt.ylabel('Predicted Fare Price')
plt.legend()
plt.show()

