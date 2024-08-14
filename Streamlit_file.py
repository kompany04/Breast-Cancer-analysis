# Importing all necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model from a pickle file
# This model was trained and saved previously
model = joblib.load('best_model_Ann.pkl')

# Load the breast cancer dataset to retrieve the feature names
breast_cancer = load_breast_cancer()
all_feature_names = breast_cancer.feature_names

# Define the indices of the top 10 features selected by SelectKBest during model training
# These indices should match those used when the model was trained
selected_feature_indices = [0, 2, 3, 7, 8, 20, 21, 23, 27, 28]  # Example indices
selected_feature_names = [all_feature_names[i] for i in selected_feature_indices]

# Set up the Streamlit app with a title
st.title('Breast Cancer Prediction App')

# Add a brief description of the app's purpose
st.write("""
This app predicts whether a breast mass is benign or malignant based on the input measurements provided.
Only the most important features, as determined by our feature selection process, are included in the prediction.
""")

# Create input fields for each of the selected features
# Users will enter the values for each feature
input_features = []
for feature in selected_feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0, format="%.6f")
    input_features.append(value)

# Make a prediction when the user clicks the "Predict" button
if st.button('Predict'):
    # Reshape the input features into the required format for model prediction
    input_array = np.array(input_features).reshape(1, -1)
    
    # Make a prediction and calculate the probabilities of each class (benign or malignant)
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)

    # Display the prediction result to the user
    st.subheader('Prediction:')
    if prediction[0] == 0:
        st.write('The breast mass is predicted to be: Benign')
    else:
        st.write('The breast mass is predicted to be: Malignant')

    # Display the prediction probabilities for each class
    st.subheader('Prediction Probability:')
    st.write(f'Benign: {probability[0][0]:.2f}')
    st.write(f'Malignant: {probability[0][1]:.2f}')

# Display the selected features used by the model, ordered by their importance
st.subheader('Feature Importance')
st.write('This model uses the following features, listed in order of importance:')
for i, feature in enumerate(selected_feature_names, 1):
    st.write(f"{i}. {feature}")
