import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("🏦 Customer Churn Prediction")
st.write("Predict whether a customer will churn or not")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("bank customer dataset.csv")
    return df

df = load_data()

# -------------------------------
# PREPROCESSING
# -------------------------------
df = df.drop(['customer_id'], axis=1)

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

df = pd.get_dummies(df, columns=['country'], drop_first=True)

X = df.drop('churn', axis=1)
y = df['churn']

# -------------------------------
# TRAIN MODEL
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# -------------------------------
# USER INPUT
# -------------------------------
st.sidebar.header("Enter Customer Details")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)
age = st.sidebar.slider("Age", 18, 80, 30)
tenure = st.sidebar.slider("Tenure", 0, 10, 3)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
products_number = st.sidebar.slider("Number of Products", 1, 4, 1)
credit_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
active_member = st.sidebar.selectbox("Is Active Member", [0, 1])
estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
country = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])

# Encode inputs
gender = 1 if gender == "Male" else 0

# Create input dataframe
input_data = pd.DataFrame({
    'credit_score': [credit_score],
    'gender': [gender],
    'age': [age],
    'tenure': [tenure],
    'balance': [balance],
    'products_number': [products_number],
    'credit_card': [credit_card],
    'active_member': [active_member],
    'estimated_salary': [estimated_salary]
})

# Add country columns
for col in X.columns:
    if 'country_' in col:
        input_data[col] = 0

if country == "Germany":
    input_data['country_Germany'] = 1
elif country == "Spain":
    input_data['country_Spain'] = 1

# Ensure column order matches training
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_data)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN")
    else:
        st.success(f"✅ Customer is likely to STAY")

    st.write(f"**Churn Probability:** {probability:.2f}")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.subheader("📊 Feature Importance")

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

st.bar_chart(importance.set_index('Feature'))
