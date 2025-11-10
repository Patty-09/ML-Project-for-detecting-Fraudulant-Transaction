import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
pickle_in = open("FDmodel.pkl","rb")
classifier=pickle.load(pickle_in)

# Prediction function
def predict_fraud(Transaction_Amount, Transaction_Type, Account_Balance,
                  Device_Type, Location, Merchant_Category, Daily_Transaction_Count,
                  Avg_Transaction_Amount_7d, Failed_Transaction_Count_7d,
                  Card_Type, Card_Age, Authentication_Method, Risk_Score):

    # Encode categorical values if needed (This assumes the model can handle them directly,or you preprocessed before training)
    input_data = np.array([[Transaction_Amount, Transaction_Type, Account_Balance,
                            Device_Type, Location, Merchant_Category, Daily_Transaction_Count,
                            Avg_Transaction_Amount_7d, Failed_Transaction_Count_7d,
                            Card_Type, Card_Age, Authentication_Method, Risk_Score]])

    prediction = model.predict(input_data)
    return prediction[0]


# Streamlit frontend
def main():
    st.markdown("""
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;"> Fraud Detection in Transactions </h2>
        </div>
    """, unsafe_allow_html=True)

    # Input fields
    Transaction_Amount = st.number_input("Transaction Amount", min_value=0.0)
    Transaction_Type = st.selectbox("Transaction Type", ['Debit', 'Credit'])
    Account_Balance = st.number_input("Account Balance", min_value=0.0)
    Device_Type = st.selectbox("Device Type", ['Mobile', 'Desktop', 'ATM', 'Other'])
    Location = st.text_input("Location (City/Code)")
    Merchant_Category = st.text_input("Merchant Category")
    Daily_Transaction_Count = st.number_input("Daily Transaction Count", min_value=0)
    Avg_Transaction_Amount_7d = st.number_input("Avg Transaction Amount (7 days)", min_value=0.0)
    Failed_Transaction_Count_7d = st.number_input("Failed Transaction Count (7 days)", min_value=0)
    Card_Type = st.selectbox("Card Type", ['Credit', 'Debit', 'Prepaid', 'Other'])
    Card_Age = st.number_input("Card Age (in months)", min_value=0)
    Authentication_Method = st.selectbox("Authentication Method", ['PIN', 'Biometric', 'OTP', 'Password'])
    Risk_Score = st.slider("Risk Score", 0, 100)

    # Prediction
    if st.button("Predict Fraud"):
        result = predict_fraud(Transaction_Amount, Transaction_Type, Account_Balance,
                               Device_Type, Location, Merchant_Category, Daily_Transaction_Count,
                               Avg_Transaction_Amount_7d, Failed_Transaction_Count_7d,
                               Card_Type, Card_Age, Authentication_Method, Risk_Score)

        if result == 1:
            st.error("Transaction is Fraudulent")
        else:
            st.success("Transaction is Legitimate")

if __name__ == '__main__':
    main()


