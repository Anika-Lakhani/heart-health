import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import OrdinalEncoder
from framingham10yr import framingham_10year_risk  # Import the function

st.header("South Asian Heart Disease Risk Calculator")

## Loading Data
df = pd.read_csv('data/heart_2020_cleaned_with_synthetic.csv')
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

import plotly.express as px
from umap import UMAP

newdf_sample = newdf.sample(1000, random_state=42)  # Take 1000 random samples
umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
data_reduced = umap.fit_transform(newdf_sample.iloc[:, :-1].values)

# UMAP for dimensionality reduction
#umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
#data_reduced = umap.fit_transform(newdf.iloc[:, :-1].values)  # Reduce features to 2D for visualization

# Add user input to the UMAP data for visualization
user_reduced = umap.transform(user.iloc[:, :-1].values)
data_reduced = np.vstack([data_reduced, user_reduced])  # Combine user data with the dataset

# Create a DataFrame for visualization
vis_df = pd.DataFrame(data_reduced, columns=['UMAP1', 'UMAP2'])
vis_df['Type'] = ['Dataset'] * (len(data_reduced) - 1) + ['User']  # Label user point

# Visualize using Plotly Express
fig = px.scatter(
    vis_df, x='UMAP1', y='UMAP2', color='Type',
    color_discrete_map={'Dataset': 'blue', 'User': 'red'},
    title='Clustering Visualization: Comparing Individual to Dataset',
    template='plotly_white'
)

fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')))
st.plotly_chart(fig)