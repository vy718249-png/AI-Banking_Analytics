import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Step 1 : Load Dataset
# ==========================================

df = pd.read_csv("loan_data.csv")

print("=" * 50)
print("First 5 Rows")
print(df.head())

print("=" * 50)
print("Dataset Info")
print(df.info())

print("=" * 50)
print("Statistics")
print(df.describe())

print("=" * 50)
print("Shape")
print(df.shape)

print("=" * 50)
print("Missing Values")
print(df.isnull().sum())

print("=" * 50)
print("Duplicate Rows")
print(df.duplicated().sum())

# ==========================================
# Step 2 : Features & Target
# ==========================================

X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"]

# ==========================================
# Step 3 : Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("=" * 50)
print("Training Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# ==========================================
# Step 4 : Build Random Forest Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==========================================
# Step 5 : Train Model
# ==========================================

model.fit(X_train, y_train)

print("=" * 50)
print("Model Training Completed")

# ==========================================
# Step 6 : Prediction
# ==========================================

y_pred = model.predict(X_test)

print("=" * 50)
print("Predictions")
print(y_pred)

# ==========================================
# Step 7 : Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("Accuracy :", accuracy)

# ==========================================
# Step 8 : Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("=" * 50)
print("Confusion Matrix")
print(cm)

# ==========================================
# Step 9 : Classification Report
# ==========================================

report = classification_report(y_test, y_pred)

print("=" * 50)
print("Classification Report")
print(report)

# ==========================================
# Step 10 : Save Model
# ==========================================

joblib.dump(model, "model.pkl")

print("=" * 50)
print("Model Saved Successfully")
print("File Name : model.pkl")

# ==========================================
# Step 11 : Test Model
# ==========================================

sample = [[30, 50000, 700, 150000]]

prediction = model.predict(sample)

print("=" * 50)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")