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
    if row["Age"] > 50:
        score += 2

    # Balance
    if row["Balance"] > 100000:
        score += 2

    # Credit Score
    if row["CreditScore"] < 500:
        score += 2

    # Activity
    if row["IsActiveMember"] == 0:
        score += 1

    # Products
    if row["NumOfProducts"] <= 1:
        score += 1

    # Exited
    if row["Exited"] == 1:
        score += 3

    # Final Risk Level
    if score >= 7:
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

# Page Configuration
st.set_page_config(page_title = "Customer Churn Risk Calculator", page_icon = "📉",
                    layout = "wide")
st.title("Customer Churn Risk Calculator")

# Input Section
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300,900,650)
    age = st.slider("Age", 18,100,35)
    balance = st.number_input("Balance", min_value = 0.0, value = 50000.0)
    estimated_salary = st.number_input("Estimated Salary", min_value = 0.0, value = 50000.0)

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

# Prediction Button
if st.button("Predict Churn Risk"):
    
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


    # Add missing columns required by model
    for col in model4.feature_names_in_:
        if col not in input_data.columns:
          input_data[col] = 0

    

    input_data = input_data[model4.feature_names_in_]

    
    st.write(input_data.T)

    probability = model4.predict_proba(input_data)[0]

    prediction = model4.predict(input_data)[0]


    class_names = model4.classes_
    probability_df = pd.DataFrame([{
    class_names[i]: round(probability[i] * 100, 2)
    for i in range(len(class_names))}])

    # Churn Probability
    predicted_index = list(class_names).index(prediction)

    # KPI Metrics
    prob_dict = {
    class_names[i]: round(probability[i] * 100, 2)
    for i in range(len(class_names))}

    # Results
    predicted_probability = round(
    prob_dict[prediction],
    2)

    st.subheader("Prediction Result")

    if prediction == "High Risk":
      st.error(
      f"⚠️ High Churn Risk ({predicted_probability}%)" )

    elif prediction == "Medium Risk":
      st.warning(
       f"⚠️ Medium Churn Risk ({predicted_probability}%)")

    else:
      st.success(
       f"✅ Low Churn Risk ({predicted_probability}%)")

    st.subheader("Risk Probabilities")

    

    # KPI Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
      st.metric(
        "Low Risk %",
        f"{prob_dict.get('Low Risk', 0)}%")

    with col2:
     st.metric(
        "Medium Risk %",
        f"{prob_dict.get('Medium Risk', 0)}%")

    with col3:
     st.metric(
        "High Risk %",
        f"{prob_dict.get('High Risk', 0)}%")
     
    predicted_probability = round(
    prob_dict[prediction],2)

   
   