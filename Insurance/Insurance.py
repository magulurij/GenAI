import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import math

# Load the dataset
file_path = r"C:\Users\srava\Downloads\Cleaned_Insurance_Dataset.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Preview:\n", df.head())

# Data Cleaning
# Checking for missing values (we expect none if previously cleaned)
print("\nMissing Values:\n", df.isnull().sum())

# Encoding categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Boxplots for each numerical column to detect outliers
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

print("\nBoxplots for Outlier Detection:")
num_columns = len(numerical_columns)
cols = 3  # Number of columns for the subplot grid
rows = math.ceil(num_columns / cols)  # Calculate number of rows needed

plt.figure(figsize=(5 * cols, 4 * rows))  # Adjust figure size based on grid size
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.show()

# Split the data into features and target variable
X = df.drop('expenses', axis=1)  # Assuming 'expenses' is the target variable
y = df['expenses']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training using RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:\nMean Squared Error (MSE): {mse:.2f}\nR-squared (R²): {r2:.2f}")

# Plotting Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='b')
plt.xlabel("Actual Expenses")
plt.ylabel("Predicted Expenses")
plt.title("Actual vs Predicted Expenses")
plt.show()
