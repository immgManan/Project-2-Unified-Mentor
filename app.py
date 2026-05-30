import streamlit as st

# Page Configuration
st.set_page_config(page_title = "Bank Churn Dashboard", layout = "wide")

# Title
st.title("Bank Customer Churn Analysis Dashboard")

# Introduction
st.markdown("""This dashboard analyzes customer churn patterns using machine learning
and business intelligence visualizations.
            Use the sidebar to navigate between pages.""")

# KPI Section
col1, col2, col3 = st.columns(3)

with col1: st.metric("Total Customers", "10,000")
with col2: st.metric("Churn Rate", "20.37%")
with col3: st.metric("Retention Rate", "79.63%")

# Divider
st.divider()

# About Project
st.subheader("Project Overview")
st.write("""This project includes:
         - Data preprocessing and feature engineering 
         - Machine learning model development and evaluation""")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.success("Select a page above")
