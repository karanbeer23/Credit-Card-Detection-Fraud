import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained TPOT model
@st.cache_resource
def load_model():
    with open('tpot_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title('Credit Card Fraud Detection')
st.write('Enter the transaction details to predict if it is fraudulent or not.')

# Create input fields for each feature
st.sidebar.header('Transaction Details')

# Assuming you have 30 features (Time, V1-V28, Amount) as in your dataframe
# For simplicity, let's create a few example input fields.
# You would need to create input fields for all 30 features (excluding 'Class')

def get_user_input():
    Time = st.sidebar.number_input('Time (seconds since first transaction)', value=0.0)
    V1 = st.sidebar.number_input('V1', value=0.0)
    V2 = st.sidebar.number_input('V2', value=0.0)
    V3 = st.sidebar.number_input('V3', value=0.0)
    V4 = st.sidebar.number_input('V4', value=0.0)
    V5 = st.sidebar.number_input('V5', value=0.0)
    V6 = st.sidebar.number_input('V6', value=0.0)
    V7 = st.sidebar.number_input('V7', value=0.0)
    V8 = st.sidebar.number_input('V8', value=0.0)
    V9 = st.sidebar.number_input('V9', value=0.0)
    V10 = st.sidebar.number_input('V10', value=0.0)
    V11 = st.sidebar.number_input('V11', value=0.0)
    V12 = st.sidebar.number_input('V12', value=0.0)
    V13 = st.sidebar.number_input('V13', value=0.0)
    V14 = st.sidebar.number_input('V14', value=0.0)
    V15 = st.sidebar.number_input('V15', value=0.0)
    V16 = st.sidebar.number_input('V16', value=0.0)
    V17 = st.sidebar.number_input('V17', value=0.0)
    V18 = st.sidebar.number_input('V18', value=0.0)
    V19 = st.sidebar.number_input('V19', value=0.0)
    V20 = st.sidebar.number_input('V20', value=0.0)
    V21 = st.sidebar.number_input('V21', value=0.0)
    V22 = st.sidebar.number_input('V22', value=0.0)
    V23 = st.sidebar.number_input('V23', value=0.0)
    V24 = st.sidebar.number_input('V24', value=0.0)
    V25 = st.sidebar.number_input('V25', value=0.0)
    V26 = st.sidebar.number_input('V26', value=0.0)
    V27 = st.sidebar.number_input('V27', value=0.0)
    V28 = st.sidebar.number_input('V28', value=0.0)
    Amount = st.sidebar.number_input('Amount', value=0.0)

    # Create a dictionary of all features
    user_data = {
        'Time': Time,
        'V1': V1,
        'V2': V2,
        'V3': V3,
        'V4': V4,
        'V5': V5,
        'V6': V6,
        'V7': V7,
        'V8': V8,
        'V9': V9,
        'V10': V10,
        'V11': V11,
        'V12': V12,
        'V13': V13,
        'V14': V14,
        'V15': V15,
        'V16': V16,
        'V17': V17,
        'V18': V18,
        'V19': V19,
        'V20': V20,
        'V21': V21,
        'V22': V22,
        'V23': V23,
        'V24': V24,
        'V25': V25,
        'V26': V26,
        'V27': V27,
        'V28': V28,
        'Amount': Amount
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

input_df = get_user_input()

st.subheader('User Input:')
st.write(input_df)

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction:')
    if prediction[0] == 0:
        st.success('Transaction is likely NOT FRAUDULENT')
    else:
        st.error('Transaction is likely FRAUDULENT')

    st.subheader('Prediction Probability:')
    st.write(f'Not Fraudulent: {prediction_proba[0][0]:.4f}')
    st.write(f'Fraudulent: {prediction_proba[0][1]:.4f}')
