import matplotlib.pyplot as plt
import pandas as pd
import pickle
import streamlit as st

# Retrain Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("dashboard_data.csv")
def risk_label(row):

    score = 0

    # Age
    if row["Age"] > 60:
        score += 1

    # Balance
    if row["Balance"] > 150000:
        score += 1

    # Credit Score
    if row["CreditScore"] < 450:
        score += 2

    # Activity
    if row["IsActiveMember"] == 0:
        score += 1

    # Products
    if row["NumOfProducts"] == 1:
        score += 1

    # Exited
    if row["Exited"] == 1:
        score += 2

    # Final Risk Level
    if score >= 6:
        return "High Risk"

    elif score >= 4:
        return "Medium Risk"

    else:
        return "Low Risk"

df["Risk Level"] = df.apply(risk_label, axis=1)
st.write(df["Risk Level"].value_counts())
# Feature Engineering
df["Balance_to_Salary_ratio"] = (
    df["Balance"] / df["EstimatedSalary"])

df["Product_Density_Indicator"] = (
    df["NumOfProducts"] / df["Age"])

df["Engagement_product_indicator"] = (
    df["NumOfProducts"] * 2
    + df["IsActiveMember"]
    + df["HasCrCard"])

df["Age_tenure_interaction"] = (
    df["Age"] * df["Tenure"])

# Features and Target
x = df.drop(["Risk Level","Exited"], axis=1)

y = df["Risk Level"]

# Encoding
x = pd.get_dummies(x, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

model4 = GradientBoostingClassifier()
model4.fit(x_train, y_train)
from sklearn.metrics import accuracy_score

preds = model4.predict(x_test)

accuracy = accuracy_score(y_test, preds)

st.write("Model Accuracy:", accuracy)

pickle.dump(model4, open("churn_model_Grad_Boost.pkl", "wb"))
model4 = pickle.load(open("churn_model_Grad_Boost.pkl", "rb"))

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
    num_of_products = st.slider("Number of Products", 0, 4, 1)
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
        'Age_tenure_interaction': [age * tenure],
        })
    # add missing columns
    for col in x.columns:
       if col not in input_data.columns:
        input_data[col] = 0
    # Match Training Columns
    input_data = input_data[x.columns]



    # Predict probabilities
    probabilities = model4.predict_proba(input_data)[0]

    # Get class names
    class_names = model4.classes_


    # Dataframe
    probability_df = pd.DataFrame({"Risk Level": class_names, "Probability (%)": [round(p*100,2) 
                                                     for p in probabilities]})
    
    # Show Table
    st.subheader("Customer Risk Probability")
    st.dataframe(probability_df, use_container_width= True)

    # Visualization
    fig, ax = plt.subplots(figsize = (7,5))
    bars = ax.bar(probability_df["Risk Level"], probability_df["Probability (%)"])

    for bar in bars:
        height = bar.get_height()
        ax.text( bar.get_x() + bar.get_width()/2, height + 1, f"{height:.1f}%",
        ha='center')

    ax.set_ylim(0,100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Customer Risk Prediction")

    st.pyplot(fig)  

    highest_risk = probability_df.loc[probability_df["Probability (%)"].idxmax(),
    "Risk Level"]

    highest_prob = probability_df["Probability (%)"].max()

    if highest_risk == "High Risk":
        st.error(f"⚠️ {highest_risk} ({highest_prob:.1f}%)")

    elif highest_risk == "Medium Risk":
        st.warning(f"⚠️ {highest_risk} ({highest_prob:.1f}%)")

    else:
        st.success(f"✅ {highest_risk} ({highest_prob:.1f}%)")