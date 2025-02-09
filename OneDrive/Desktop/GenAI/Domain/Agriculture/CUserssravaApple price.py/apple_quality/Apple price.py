import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# Load your dataset (update the path if necessary)
file_path = r"C:\Users\srava\Downloads\apple_quality.csv"
df = pd.read_csv(file_path)

# Columns of interest
columns = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity','Quality']

# 1. Handling missing values: Identifying missing values in each column
missing_values = df[columns].isnull().sum()
print("Missing values in each column:\n", missing_values)

# 2. Identifying rows where all values are zero in the selected columns
rows_all_zero = df.loc[(df[columns] == 0).all(axis=1)]
print("\nRows where all values are zero:\n", rows_all_zero)

# 3. Providing descriptive statistics for the selected columns
descriptive_stats = df[columns].describe()
print("\nDescriptive statistics for the selected columns:\n", descriptive_stats)

# Create subplots for each column
columns = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity','Quality']

# Creating boxplots for the specified columns
plt.figure(figsize=(10, 6))  # Set the figure size

# Create boxplot
df[columns].boxplot()

# Set title and labels
plt.title('Boxplot for Apple Quality Features')
plt.ylabel('Values')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the plot
plt.tight_layout()
plt.show()

df['Quality'] = df['Quality'].map({'good': 1, 'bad': 0})


# 4. Creating a correlation matrix
correlation_matrix = df[columns].corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# 5. Visualizing the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Matrix for Apple Quality Features')
plt.show()

# Features and target variable
X = df[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']]  # Update with your feature columns
y = df['Quality']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)


# Print the results
print(f"\nAccuracy: {accuracy:.2f}")