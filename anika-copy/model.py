# Add error handling for Google Drive mounting
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import os
import sys

if IN_COLAB:
    # Mount Google Drive
    drive.mount('/content/drive')

    # Set the working directory to your project folder
    PROJECT_PATH = '/content/drive/MyDrive/anika-copy'
    os.chdir(PROJECT_PATH)

    # Add project directory to Python path
    if PROJECT_PATH not in sys.path:
        sys.path.append(PROJECT_PATH)

# Rest of your imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import OrdinalEncoder
from framingham10yr import framingham_10year_risk
import plotly.express as px
from umap import UMAP

# Add error handling for data loading
try:
    df = pd.read_csv(os.path.join('data', 'heart_2020_cleaned_with_synthetic.csv'))
except FileNotFoundError:
    st.error("Error: Could not find the data file. Please check the path and file existence in Google Drive.")
    st.stop()

# Cache the model training to improve performance
@st.cache_data
def train_model(param, target):
    model = RandomForestClassifier()
    model.fit(param, target)
    return model

# Cache the UMAP transformation
@st.cache_data
def perform_umap_transform(data):
    umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    return umap.fit_transform(data)

st.header("South Asian Heart Disease Risk Calculator")

## Loading Data
newdf = df.copy()

## Function to Collect User Inputs
def user_input_features():
    st.write("**Please fill out the questionnaire below to see if you are at risk of heart disease:**")
    
    BMI = st.slider("1. Enter BMI:", 0.0, 110.0, 25.0)
    smoker = st.selectbox("2. Have you smoked over 100 cigarettes in your lifetime?", ["Yes", "No"])
    alc = st.selectbox("3. Are you a heavy drinker?", ["Yes", "No"])
    stroke = st.selectbox("4. Have you ever had a stroke?", ["Yes", "No"])
    physical = st.slider("5. Bad physical health days in the last 30 days:", 0, 30, 5)
    mental = st.slider("6. Bad mental health days in the last 30 days:", 0, 30, 5)
    climb = st.selectbox("7. Do you have difficulty climbing stairs?", ["Yes", "No"])
    sex = st.selectbox("8. What is your sex?", ["Male", "Female"])
    age = st.slider("9. Enter your age:", 20, 79, 50)
    diabetes = st.selectbox("10. Have you ever been told you are diabetic?", ["Yes", "No"])
    exercise = st.selectbox("11. Have you exercised in the past 30 days?", ["Yes", "No"])
    sleep = st.slider("12. How much do you sleep per day (hours)?", 0, 24, 7)
    gen_health = st.selectbox("13. How would you rate your general health?", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
    asthma = st.selectbox("14. Have you ever been told you have asthma?", ["Yes", "No"])
    kidney = st.selectbox("15. Have you ever been told you have kidney disease?", ["Yes", "No"])
    cancer = st.selectbox("16. Have you ever been told you have skin cancer?", ["Yes", "No"])

    # Framingham-specific inputs
    total_cholesterol = st.slider("17. Enter your total cholesterol (mg/dL):", 130, 320, 200)
    hdl_cholesterol = st.slider("18. Enter your HDL cholesterol (mg/dL):", 20, 100, 50)
    systolic_bp = st.slider("19. Enter your systolic blood pressure (mm Hg):", 90, 200, 120)
    bp_med = st.selectbox("20. Are you taking blood pressure medications?", ["Yes", "No"])
    
    # Convert Yes/No to 1/0 for Framingham function and model inputs

    data = {
        'BMI': BMI, 'Smoking': smoker, 'AlcoholDrinking': alc, 'Stroke': stroke,
        'PhysicalHealth': physical, 'MentalHealth': mental, 'DiffWalking': climb,
        'Sex': sex, 'Age': age, 'Diabetic': diabetes, 'PhysicalActivity': exercise,
        'GenHealth': gen_health, 'SleepTime': sleep, 'Asthma': asthma,
        'KidneyDisease': kidney, 'SkinCancer': cancer, 'Race': 'Asian',
        'TotalCholesterol': total_cholesterol, 'HDLCholesterol': hdl_cholesterol,
        'SystolicBP': systolic_bp, 'BloodPressureMedication': bp_med
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

## Collect Inputs
user = user_input_features()

discrete = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'Diabetic',
            'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer', 'BloodPressureMedication']

from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()
newdf[discrete] = enc.fit_transform(newdf[discrete])

# Define columns to encode
object_columns = ['HeartDisease', 'AgeCategory', 'Race']

# Initialize the encoder
enc = OrdinalEncoder()

# Fit and transform the object columns in newdf
newdf[object_columns] = enc.fit_transform(newdf[object_columns])

# List of object columns to encode
object_columns_2 = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 
                  'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 
                  'KidneyDisease', 'SkinCancer', 'BloodPressureMedication']

# Initialize and fit the encoder (using newdf to maintain consistency)
user[object_columns_2] = enc.fit_transform(user[object_columns_2])  # Fit on the original dataset

# Check the transformed user DataFrame
st.write("User input after encoding:")
st.write(user)

st.write("Column data types in newdf:")
st.write(newdf.dtypes)

st.write("Data types in user:")
st.write(user.dtypes)

'''
## Train Models
param = newdf.iloc[:, :-1].values
target = newdf.iloc[:, -1].values

model = RandomForestClassifier()
model.fit(param, target)
prediction = model.predict(user)

st.subheader("Prediction using RandomForestClassifier:")
st.write("Chances of Heart Disease:", "Yes" if prediction[0] == 1 else "No")

## Calculate Framingham Risk Score
framingham_risk = framingham_10year_risk(
    sex=user['Sex'][0], age=user['Age'][0],
    total_cholesterol=user['TotalCholesterol'][0],
    hdl_cholesterol=user['HDLCholesterol'][0],
    systolic_blood_pressure=user['SystolicBP'][0],
    smoker=user['Smoking'][0],
    blood_pressure_med_treatment=user['BloodPressureMedication'][0],
    race='south_asian'
)

st.subheader("Framingham 10-Year Heart Disease Risk Score:")
st.write(framingham_risk)
'''

## UMAP Visualization
# Add error handling for visualizations
try:
    # Prepare data for UMAP
    # First, ensure all categorical columns are encoded
    categorical_columns = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 
                         'Sex', 'Diabetic', 'PhysicalActivity', 'GenHealth', 
                         'Asthma', 'KidneyDisease', 'SkinCancer', 'Race', 
                         'BloodPressureMedication']
    
    # Create copy of data for UMAP
    umap_df = newdf.copy()
    user_umap = user.copy()
    
    # Set Race column for user data (it's always 'Asian' in your case)
    user_umap['Race'] = 'Asian'
    
    # Encode categorical columns
    enc = OrdinalEncoder()
    umap_df[categorical_columns] = enc.fit_transform(umap_df[categorical_columns])
    
    # Transform user data using the same encoder
    for col in categorical_columns:
        # Reshape for single sample
        user_val = user_umap[col].values.reshape(-1, 1)
        user_umap[col] = enc.transform(user_val).ravel()
    
    # Sample data and perform UMAP
    newdf_sample = umap_df.sample(1000, random_state=42)
    data_reduced = perform_umap_transform(newdf_sample.iloc[:, :-1].values)
    
    # Transform user data
    user_reduced = perform_umap_transform(user_umap.iloc[:, :-1].values)
    
    # Combine data for visualization
    data_reduced = np.vstack([data_reduced, user_reduced])
    
    # Create visualization DataFrame
    vis_df = pd.DataFrame(data_reduced, columns=['UMAP1', 'UMAP2'])
    vis_df['Type'] = ['Dataset'] * (len(data_reduced) - 1) + ['User']
    
    # Create plot
    fig = px.scatter(
        vis_df, x='UMAP1', y='UMAP2', color='Type',
        color_discrete_map={'Dataset': 'blue', 'User': 'red'},
        title='Clustering Visualization: Comparing Individual to Dataset',
        template='plotly_white'
    )
    
    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')))
    st.plotly_chart(fig)

except Exception as e:
    st.error(f"Error generating UMAP visualization: {str(e)}")
    st.write("Debug info:")
    st.write("Data types in user DataFrame:", user_umap.dtypes)
    st.write("Data types in newdf DataFrame:", umap_df.dtypes)
    st.write("Categorical columns:", categorical_columns)
    st.write("User data head:", user_umap.head())

## Slider Visualization
# Create uncertainty visualization with sliders
st.subheader("Risk Factor Sensitivity Analysis")
st.write("See how changing key factors affects your heart disease risk prediction")

# Select key features for sensitivity analysis
key_features = ['BMI', 'SystolicBP', 'TotalCholesterol', 'HDLCholesterol']
original_values = user[key_features].iloc[0].copy()
sensitivity_results = []

# Create columns for side-by-side visualization
col1, col2 = st.columns([1, 2])

with col1:
    # Feature selector
    selected_feature = st.selectbox(
        "Select feature to analyze:",
        key_features
    )
    
    # Get feature range (we should probably define this manually I'm guessing?)
    feature_min = user[selected_feature].iloc[0] * 0.5
    feature_max = user[selected_feature].iloc[0] * 1.5
    
    # Create slider for selected feature
    test_value = st.slider(
        f"Adjust {selected_feature}",
        float(feature_min),
        float(feature_max),
        float(original_values[selected_feature]),
        step=0.1
    )

with col2:
    # Create test cases
    test_range = np.linspace(feature_min, feature_max, 50)
    probabilities = []
    
    # Calculate probabilities for each test value
    for value in test_range:
        test_user = user.copy()
        test_user[selected_feature] = value
        prob = framingham_10year_risk(
            sex=test_user['Sex'][0], 
            age=test_user['Age'][0],
            race='south_asian',
            total_cholesterol=test_user['TotalCholesterol'][0],
            hdl_cholesterol=test_user['HDLCholesterol'][0],
            systolic_blood_pressure=test_user['SystolicBP'][0],
            smoker=test_user['Smoking'][0],
            blood_pressure_med_treatment=test_user['BloodPressureMedication'][0]
        )['percent_risk']

        # Convert percentage string to float (removing % sign and handling '<' symbol)
        if isinstance(prob, str):
            prob = float(prob.replace('%', '').replace('<', '')) / 100

        probabilities.append(prob)
    
    # Create sensitivity plot
    fig = px.line(
        x=test_range, 
        y=probabilities,
        labels={
            'x': selected_feature,
            'y': 'Heart Disease Probability'
        },
        title=f'Risk Sensitivity to {selected_feature}'
    )
    
    # Add vertical line for current value
    fig.add_vline(
        x=test_value,
        line_dash="dash",
        line_color="red",
        annotation_text="Current Value"
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig)

# Show risk at current value
test_user = user.copy()
test_user[selected_feature] = test_value
current_risk = framingham_10year_risk(
    sex=test_user['Sex'][0], 
    age=test_user['Age'][0],
    race='south_asian',
    total_cholesterol=test_user['TotalCholesterol'][0],
    hdl_cholesterol=test_user['HDLCholesterol'][0],
    systolic_blood_pressure=test_user['SystolicBP'][0],
    smoker=test_user['Smoking'][0],
    blood_pressure_med_treatment=test_user['BloodPressureMedication'][0]
)['percent_risk']

if isinstance(current_risk, str):
    current_risk = float(current_risk.replace('%', '').replace('<', '')) / 100

st.write(f"Risk probability at selected value: {current_risk:.2%}")