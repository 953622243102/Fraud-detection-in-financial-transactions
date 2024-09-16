import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle  # Import the pickle module

# Load and display the data
data = pd.read_csv("dataset.csv")
print(data.head())

# Checking value counts for transaction types
type_counts = data["type"].value_counts()
transactions = type_counts.index
quantity = type_counts.values

# Checking correlation with isFraud
# Exclude non-numeric columns for correlation calculation
correlation = data.select_dtypes(include=[np.number]).corr()

# Display the correlation with the target column 'isFraud'
print(correlation["isFraud"].sort_values(ascending=False))


# Map the type to numerical values for model training
data["type"] = data["type"].map({
    "CASH_OUT": 1, "PAYMENT": 2, 
    "CASH_IN": 3, "TRANSFER": 4,
    "DEBIT": 5
})

# Ensure the target variable is numeric (0 and 1)
data["isFraud"] = data["isFraud"].map({0: 0, 1: 1})  # Keep it binary
print(data.head())

# Splitting the data for training
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data["isFraud"])  # Target needs to be 1D for training

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)

# Training the DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

# Model evaluation
print("Model accuracy:", model.score(xtest, ytest))

# Save the trained model as a .pkl file
with open("fraud_detection_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Prediction on new data
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print("Prediction (Fraud = 1, No Fraud = 0):", model.predict(features))
