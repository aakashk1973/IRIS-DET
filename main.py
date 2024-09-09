# main.py
import streamlit as st
import pickle
import numpy as np

# Load the trained SVM model from the pickle file
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Streamlit App Title
st.title('SVM Model Prediction App')

# Instructions
st.write("Enter the values for the 4 features to make a prediction:")

# Input fields for the 4 features
feature1 = st.number_input('Feature 1', value=0.0)
feature2 = st.number_input('Feature 2', value=0.0)
feature3 = st.number_input('Feature 3', value=0.0)
feature4 = st.number_input('Feature 4', value=0.0)

# Button to trigger prediction
if st.button('Make Prediction'):
    # Prepare the features for prediction
    features = np.array([[feature1, feature2, feature3, feature4]])

    # Make prediction using the loaded SVM model
    prediction = svm_model.predict(features)

    # Display the prediction result
    st.write(f'The predicted class is: {prediction[0]}')

