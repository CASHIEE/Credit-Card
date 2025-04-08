# Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
credit_card_data = pd.read_csv("creditcard.csv")

# Display basic info about the dataset
print("First 5 rows:\n", credit_card_data.head())
print("\nLast 5 rows:\n", credit_card_data.tail())
print("\nDataset Info:")
credit_card_data.info()

# Check for missing values
print("\nMissing values in each column:\n", credit_card_data.isnull().sum())

# Distribution of classes
print("\nClass distribution:\n", credit_card_data['Class'].value_counts())

# Separate legitimate and fraudulent transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print("\nShape of legit transactions:", legit.shape)
print("Shape of fraudulent transactions:", fraud.shape)

# Statistical analysis
print("\nLegit transaction amount stats:\n", legit.Amount.describe())
print("\nFraudulent transaction amount stats:\n", fraud.Amount.describe())

# Mean comparison
print("\nMean values grouped by class:\n", credit_card_data.groupby('Class').mean())

# Under-Sampling
legit_sample = legit.sample(n=len(fraud), random_state=1)  # Ensure reproducibility

# Create a new balanced dataset
new_dataset = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=1)  # Shuffle the dataset

# Verify new class distribution
print("\nNew dataset class distribution:\n", new_dataset['Class'].value_counts())

# Mean comparison of the new dataset
print("\nNew dataset mean values grouped by class:\n", new_dataset.groupby('Class').mean())

# Splitting the data into features and labels
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print("\nShapes:")
print("Full Feature Set:", X.shape)
print("Training Set:", X_train.shape)
print("Testing Set:", X_test.shape)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)  # Ensure convergence
model.fit(X_train, Y_train)

# Model Evaluation
X_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, X_train_pred)
print("\nAccuracy on Training data:", train_accuracy)

X_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_pred)
print("Accuracy on Test data:", test_accuracy)
