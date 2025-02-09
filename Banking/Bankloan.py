# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Dataset
file_path = r'C:\Users\srava\Downloads\archive (12)\Loan_default.csv'
df = pd.read_csv(file_path)

# Drop non-numeric columns to prepare for correlation analysis
# This includes columns like 'LoanID' if it exists
df_numeric = df.select_dtypes(include=[np.number])

# Exploratory Data Analysis (EDA)

# Distribution of Variables
df_numeric.hist(figsize=(12, 10), bins=30)
plt.suptitle('Distribution of Features')
plt.show()

# Correlation Heatmap (only numeric columns)
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Identifying Outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_numeric)
plt.xticks(rotation=90)
plt.title('Outliers in Numerical Columns')
plt.show()

# Splitting the Dataset into Features and Target
# Assuming 'Default' is the target column.
X = df_numeric.drop('Default', axis=1, errors='ignore')  # Features
y = df_numeric['Default']                                # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the Model using Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Making Predictions
y_pred = model.predict(X_test_scaled)

# Evaluating the Model
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualizing Important Features
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()



