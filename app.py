import streamlit as st
import pickle
import pandas as pd


# load model

with open("churn_pred.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction")

st.write("Enter Customer Details:")

# inputs 
call_failure = st.number_input("Call  Failure", min_value=0)
complains = st.selectbox("Complains",[0,1])
sub_length = st.number_input("Subscription  Length", min_value=0)
charge_amount = st.number_input("Charge  Amount", min_value=0.0)
seconds_use = st.number_input("Seconds of Use", min_value=0.0)
freq_use = st.number_input("Frequency of use", min_value=0)
freq_sms = st.number_input("Frequency of SMS", min_value=0.0)
distinct_calls = st.number_input("Distinct Called Numbers", min_value=0)
age_group = st.selectbox("Age Group", [1, 2, 3])
tariff_plan = st.selectbox("Tariff Plan", [1, 2, 3])
status = st.selectbox("Status", [1, 2])
age = st.number_input("Age", min_value=0)
cust_value = st.number_input("Customer Value", min_value=0.0)

# prediction button
if st.button("Predict Churn"):
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

    if pred == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")