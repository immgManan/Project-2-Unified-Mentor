import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt

# Retrain Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(page_title = "Customer Churn Risk Calculator", page_icon = "📉",
                    layout = "wide")
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

y = df["Exited"]

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
    num_of_products = st.slider("Number of Products", 0, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
    is_active_member = st.selectbox("Is Active Member", [0, 1])

# Encoding
gender = 1 if gender == "Male" else 0

geo_spain = 1 if geography == "Spain" else 0
geo_germany = 1 if geography == "Germany" else 0

# Prediction Button
# Keep prediction state
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

if st.button("Predict Churn Risk"):
    st.session_state.predict_clicked = True

if st.session_state.predict_clicked:
    
    input_data = pd.DataFrame({
     'Year': [2025],
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
    # Prediction
    prediction = model4.predict(input_data)[0]

    # Probability
    probability = model4.predict_proba(input_data)[0]

    retention_probability = round(probability[0]*100, 2)
    churn_probability = round(probability[1]*100, 2)    

    # Results
  

    st.subheader("Prediction Result")
    if prediction == 1:
       st.error(f"⚠️ Customer Likely to Churn ({churn_probability}%)")
    else:
       st.success(f"✅ Customer Likely to Stay ({retention_probability}%)")   

    st.subheader("Risk Probabilities")

    # KPI Metrics
    col1, col2 = st.columns(2)

    with col1:
      st.metric("Retention Probability", f"{retention_probability}")

    with col2:
     st.metric("Churn Probability", f"{churn_probability}")

    # WHAT IF SCENARIO SIMULATOR
    st.subheader("What-if Scenario Simulator")
    st.write("Compare Original vs Modified Customer Risk")

    # Original Customer prediction
    original_input = pd.DataFrame({
        'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Balance_to_Salary_ratio' : [balance/estimated_salary if estimated_salary > 0 else 0],
    'Age_tenure_interaction' : [age*tenure]
    })

    # Match Columns
    for col in x.columns:
        if col not in original_input.columns:
            original_input[col]=0
    original_input = original_input[x.columns]

    # Original Prediction
    original_prob = model4.predict_proba(original_input)[0][1]*100

    # Form Start
    with st.form("what-if-form"):

      # Modified Scenario
      st.markdown("Modify Customer Scenario")

      col3, col4 = st.columns(2)
      with col3:
        new_balance = st.slider("New Balance", 0,250000,int(balance))
        new_products = st.slider("New Number Of Products", 0,4,int(num_of_products))

      with col4:
        new_active = st.selectbox("New Active Status", [0,1])
        new_credit_score = st.slider("New Credit Score", 300,900, int(credit_score))
      
      # keep simulator state
      if "run_simulation" not in st.session_state:
        st.session_state.run_simulation = False

      simulate_button = st.form_submit_button("Run What-If-Simulation")

      if simulate_button:
        st.session_state.run_simulation = True

    # Run only after button click
    if st.session_state.run_simulation:  
      # Modified Customer Data
      modified_input = pd.DataFrame({'CreditScore': [new_credit_score],
      'Age': [age],
      'Tenure': [tenure],
      'Balance': [new_balance],
      'NumOfProducts': [new_products],
      'HasCrCard': [has_cr_card],
      'IsActiveMember': [new_active],
      'EstimatedSalary': [estimated_salary],
      'Balance_to_Salary_ratio' : [balance/estimated_salary],
      'Age_tenure_interaction' : [age*tenure]
      })

      for col in x.columns:
        if col not in modified_input.columns:
            modified_input[col] = 0

      modified_input = modified_input[x.columns]

      modified_prob = model4.predict_proba(modified_input)[0][1] * 100

      # Comparison Table
      comparison_df = pd.DataFrame({"Scenario":["Original Customer", "Modified Customer"],
                                  "Churn Probability (%)": [round(original_prob,2), 
                                                            round(modified_prob,2)]})
    
      st.subheader("Scenario Comparison")

      st.dataframe(comparison_df, use_container_width=True)

      # Visualization
      fig2, ax2 = plt.subplots(figsize = (7,5))

      bars = ax2.bar(comparison_df["Scenario"], comparison_df["Churn Probability (%)"])

      for bar in bars:
        height = bar.get_height()
        ax2.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{height:.1f}%",
        ha='center')

      ax2.set_ylim(0,100)

      ax2.set_ylabel("Churn Probability (%)")
      ax2.set_title("What-if Scenario Comparison")
      st.pyplot(fig2)

      # Final Insight
      difference = round(original_prob - modified_prob,2)

      if modified_prob < original_prob:
        st.success(f"✅ Risk Reduced by {difference}%")

      elif modified_prob > original_prob:
        st.error(f"⚠️ Risk Increased by {abs(difference)}%")
      else:
        st.info("No Change In Risk")