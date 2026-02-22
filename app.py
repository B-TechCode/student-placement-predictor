import streamlit as st
import pickle
import numpy as np

# Page Config
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}
.result-success {
    background-color: #0f5132;
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 20px;
}
.result-fail {
    background-color: #842029;
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.markdown('<p class="big-title"> Student Placement Predictor</p>', unsafe_allow_html=True)
st.write("Enter student details below to predict placement status.")

st.divider()

# Inputs
col1, col2 = st.columns(2)

with col1:
    iq = st.number_input("Enter IQ", min_value=50, max_value=200, value=100)

with col2:
    cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, value=8.0)

st.divider()

if st.button(" Predict Placement"):
    input_data = np.array([[cgpa, iq]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.markdown('<div class="result-success"> Student WILL be Placed</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-fail"> Student will NOT be Placed</div>', unsafe_allow_html=True)

st.divider()
st.caption("Built with using Streamlit & Machine Learning")