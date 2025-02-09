import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the cleaned dataset
data = pd.read_csv(r'C:\Users\srava\Downloads\updated_ecommerce_dataset (1).csv')

# Print column names to identify the correct target column
print("Columns in the dataset:")
print(data.columns)

# Step 1: Define Features and Target
X = data.drop('Purchased', axis=1)  # Features
y = data['Purchased']  # Target variable

# Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Step 2: Data Cleaning Checks
print("\nMissing values in the dataset:")
print(data.isnull().sum())
print("\nDuplicate rows in the dataset:")
print(data.duplicated().sum())

# Step 3: Outlier Detection with Box Plots
numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=X[column])
    plt.title(f'Box Plot for {column}')
    plt.show()

# Step 4: EDA - Histograms and Correlation Heatmap
X[numerical_columns].hist(bins=15, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Features', fontsize=16)
plt.show()

sns.pairplot(X)
plt.suptitle('Pair Plot of Features', fontsize=16)
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(X[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Step 5: Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Initialize and Train the Random Forest Classifier Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Predictions and Evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
classification_rep = classification_report(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)

print("\n--- Purchase Prediction Results ---")
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)

# Step 8: Visualization of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Purchase', 'Purchase'], yticklabels=['No Purchase', 'Purchase'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
