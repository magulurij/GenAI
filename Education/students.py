# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Cleaned Dataset
file_path = r'C:\Users\srava\Downloads\Student_performance_data _.csv'
df = pd.read_csv(file_path)

# Initial Data Overview
print("Dataset Info:")
print(df.info())
print("\nDataset Head:")
print(df.head())

# Step 1: Data Cleaning
# Check for any missing values
print("\nMissing Values in Dataset:")
print(df.isnull().sum())

# Filling missing values with the median (for example) if any exist
df.fillna(df.median(numeric_only=True), inplace=True)

# Step 2: Identifying Outliers
# Box plots for outlier detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=[np.number]))
plt.xticks(rotation=90)
plt.title('Outliers in Numerical Columns')
plt.show()

# Step 3: Exploratory Data Analysis (EDA)
# Distribution of Variables
df.hist(figsize=(12, 10), bins=30)
plt.suptitle('Distribution of Features')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 4: Splitting the Dataset into Features and Target
# Assuming 'Performance' is the target column. Replace with actual target if different
X = df.drop('Performance', axis=1)  # Features
y = df['Performance']               # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Training the Model
# Using Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 7: Making Predictions
y_pred = model.predict(X_test_scaled)

# Step 8: Evaluating the Model
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

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
