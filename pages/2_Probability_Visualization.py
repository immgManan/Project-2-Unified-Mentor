import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ML Libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Page Config
st.set_page_config(page_title = "Probability Visualization", layout = "wide")
st.title("Churn Probability Visualization")

# Load dataset
df2 = pd.read_csv("dashboard_data.csv")

# Risk Label Creation
def risk_label(row):
    score = 0
    # Age
    if row["Age"] >50:
        score += 2
    # Balance
    if row["Balance"] > 100000:
        score += 2
    # Credit Score
    if row["CreditScore"] < 500:
        score += 2
    if row["IsActiveMember"] == 0:
        score += 1
    if row["NumOfProducts"] <= 1:
        score += 1
    if row["Exited"] == 1:
        score += 3

    if score >= 7:
        return "High Risk"
    elif score >= 4:
        return "Medium Risk"
    else:
        return "Low Risk"

df2["Risk Level"] = df2.apply(risk_label, axis=1)

# Features and Target
x = df2.drop(["Risk Level", "Exited"], axis = 1)
y = df2["Risk Level"]

# Encoding
x = pd.get_dummies(x, drop_first = True)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)

# Train Model
model = GradientBoostingClassifier()
model.fit(x_train, y_train)

# Input Customer Features
st.subheader("Input Customer Features")

col1, col2 = st.columns(2)
with col1:
    credit_score = st.slider("Credit Score", 300,900,650)
    age = st.slider("Age", 18,100,35)
    balance = st.number_input("Balance", min_value =0, value = 50000)
    estimated_salary = st.number_input("Estimated Salary", min_value = 0, value = 50000)
with col2:
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male","Female"])
    tenure = st.slider("Tenure", 0, 10, 5)
    num_of_products = st.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
    is_active_member = st.selectbox("Is Active Member", [0, 1])

# Encoding
gender = 1 if gender == "Male" else 0

geo_spain = 1 if geography == "Spain" else 0
geo_germany = 1 if geography == "Germany" else 0

# Predict Button
if st.button("Visualize Churn Probability"):
    input_data = pd.DataFrame({
     'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
        'Gender': [gender],
        'Geography_Germany': [geo_germany],
        'Geography_Spain': [geo_spain],
        'Balance_to_Salary_ratio': [balance / estimated_salary if estimated_salary > 0 else 0],
        'Product_Density_Indicator': [num_of_products / age if age > 0 else num_of_products],
        'Engagement_product_indicator': [num_of_products*2 + is_active_member + has_cr_card],
        'Age_tenure_interaction': [age * tenure],
        })
    # add missing columns
    for col in x.columns:
       if col not in input_data.columns:
        input_data[col] = 0
    # Match Training Columns
    input_data = input_data[x.columns]



    # Predict probabilities
    probabilities = model.predict_proba(input_data)[0]

    class_names = model.classes_

    # Probability DataFarme
    probability_df = pd.DataFrame({"Risk Level": class_names , "Probability (%)": [round(p*100,2) for p
                                 in probabilities]})


# Show Table 
    st.subheader("Customer Risk Probabilities")
    st.dataframe(probability_df, use_container_width=True)

# Visualization
    fig, ax = plt.subplots(figsize = (8,5))
    ax.bar(probability_df["Risk Level"],probability_df["Probability (%)"])

    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Risk Level")

    ax.set_title("Customer Churn Probability Distribution")
    st.pyplot(fig)
