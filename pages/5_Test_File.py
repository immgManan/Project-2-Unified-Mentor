import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt

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
x = df.drop(["Risk Level","Exited","Geography_Germany","Geography_Spain" ], axis=1)

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

# Feature Importance Dashboard
st.subheader("Feature Importance Dashboard")

# Importance DataFrame
importance_df = pd.DataFrame({"Feature": x.columns, "Importance": model4.feature_importances_})

# Sort Values
importance_df = importance_df.sort_values(by = "Importance", ascending = False)

# show Table
st.dataframe(importance_df, use_container_width=True)

# Plot
fig, ax = plt.subplots(figsize = (10,6))

ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.invert_yaxis()

ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")

ax.set_title("Feature Importance")
st.pyplot(fig)


# SHAP value analysis
import shap
import numpy as np

st.subheader("SHAP Value Analysis")

# Create SHAP explainer
explainer = shap.TreeExplainer(model4)

sample_x = x_test.sample(100, random_state=42)

shap_values = explainer.shap_values(sample_x)

# SHAP detailed impact plot
st.subheader("SHAP Feature Impact Distribution")

fig2, ax2 = plt.subplots(figsize=(10,6))

shap.summary_plot(shap_values, sample_x, show= False)
st.pyplot(fig2)

# Top SHAP features Table
st.subheader("Top SHAP Features")

shap_importance = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({"Feature": sample_x.columns, "SHAP Importance": shap_importance})

# sort
shap_df = shap_df.sort_values(by = "SHAP Importance", ascending = False)

# Display
st.dataframe(shap_df, use_container_width=True)
