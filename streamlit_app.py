import streamlit as st


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load or create your dataset
data = {
    "Sugar_Level": [90, 150, 110, 130, 80, 145, 95, 160, 100, 140],
    "BP_Systolic": [120, 130, 125, 135, 110, 140, 120, 145, 125, 138],
    "BP_Diastolic": [80, 85, 80, 90, 70, 95, 80, 100, 80, 92],
    "Weight": [65, 72, 70, 75, 60, 80, 68, 85, 72, 78],
    "Hemoglobin": [14.2, 13.4, 13.8, 14.5, 12.1, 12.8, 15.0, 13.0, 14.0, 13.7],
    "Age": [25, 30, 22, 28, 35, 50, 29, 40, 23, 32],
    "Donation": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Split the data
X = df.drop(columns=["Donation"])
y = df["Donation"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Streamlit UI
st.title("Blood Donation Eligibility Predictor")

st.write("""
### Enter the following health details to predict if someone is eligible to donate blood:
""")

# Input fields
sugar_level = st.number_input("Sugar Level (mg/dL)", min_value=50, max_value=200, value=100)
bp_systolic = st.number_input("BP Systolic (mmHg)", min_value=80, max_value=180, value=120)
bp_diastolic = st.number_input("BP Diastolic (mmHg)", min_value=50, max_value=120, value=80)
weight = st.number_input("Weight (kg)", min_value=30, max_value=150, value=70)
hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.5)
age = st.number_input("Age (years)", min_value=18, max_value=100, value=25)

# Make a prediction
if st.button("Predict"):
    input_data = np.array([[sugar_level, bp_systolic, bp_diastolic, weight, hemoglobin, age]])
    prediction = rf.predict(input_data)
    result = "Eligible to Donate" if prediction[0] == 1 else "Not Eligible to Donate"
    st.write(f"### Prediction: {result}")

# Show model accuracy
if st.checkbox("Show Model Accuracy"):
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2%}")
