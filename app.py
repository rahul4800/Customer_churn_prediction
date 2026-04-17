

j
import streamlit as st
import pickle
import pandas as pd

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load model
with open("churn_pred.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("📊 Customer Churn Prediction System")
st.markdown("Predict whether a telecom customer is likely to **churn or stay** based on their usage behavior.")

st.divider()

# Sidebar for inputs
st.sidebar.header("📋 Customer Information")

call_failure = st.sidebar.number_input("Call Failure", min_value=0)
complains = st.sidebar.selectbox("Complains", [0,1])
sub_length = st.sidebar.number_input("Subscription Length", min_value=0)
charge_amount = st.sidebar.number_input("Charge Amount", min_value=0.0)
seconds_use = st.sidebar.number_input("Seconds of Use", min_value=0.0)
freq_use = st.sidebar.number_input("Frequency of Use", min_value=0)
freq_sms = st.sidebar.number_input("Frequency of SMS", min_value=0.0)
distinct_calls = st.sidebar.number_input("Distinct Called Numbers", min_value=0)
age_group = st.sidebar.selectbox("Age Group", [1,2,3])
tariff_plan = st.sidebar.selectbox("Tariff Plan", [1,2,3])
status = st.sidebar.selectbox("Status", [1,2])
age = st.sidebar.number_input("Age", min_value=0)
cust_value = st.sidebar.number_input("Customer Value", min_value=0.0)

st.subheader("Customer Input Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Call Failure", call_failure)
    st.metric("Complains", complains)
    st.metric("Subscription Length", sub_length)

with col2:
    st.metric("Charge Amount", charge_amount)
    st.metric("Seconds of Use", seconds_use)
    st.metric("Frequency of Use", freq_use)

with col3:
    st.metric("Customer Value", cust_value)
    st.metric("Age", age)
    st.metric("Distinct Calls", distinct_calls)

st.divider()

# Prediction button
if st.button("🔍 Predict Customer Churn"):

    data = pd.DataFrame([[ 
        call_failure, complains, sub_length, charge_amount,
        seconds_use, freq_use, freq_sms, distinct_calls,
        age_group, tariff_plan, status, age, cust_value
    ]],
    columns=[
        "Call  Failure",
        "Complains",
        "Subscription  Length",
        "Charge  Amount",
        "Seconds of Use",
        "Frequency of use",
        "Frequency of SMS",
        "Distinct Called Numbers",
        "Age Group",
        "Tariff Plan",
        "Status",
        "Age",
        "Customer Value"
    ])

    pred = model.predict(data)[0]

    st.divider()

    st.subheader("Prediction Result")

    if pred == 1:
        st.error("⚠️ This customer is **likely to churn**.")
    else:
        st.success("✅ This customer is **likely to stay**.")

st.divider()

st.caption("Machine Learning Model: Random Forest | Built with Streamlit")