import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset and model (using pre-trained model or training a new model)
def load_data():
    # Synthetic dataset generation (from previous code)
    np.random.seed(42)
    plant_types = ["Wheat", "Rice", "Tomato", "Corn", "Potato"]
    regions = ["North", "South", "East", "West"]
    soil_types = ["Sandy", "Loamy", "Clay", "Silty"]
    disease_types = ["Fungal", "Bacterial", "Viral", "None"]

    num_rows = 1000
    data = {
        "Plant Type": np.random.choice(plant_types, num_rows),
        "Region": np.random.choice(regions, num_rows),
        "Soil Type": np.random.choice(soil_types, num_rows),
        "Temperature (C)": np.random.uniform(15, 40, num_rows),
        "Humidity (%)": np.random.uniform(30, 90, num_rows),
        "Rainfall (mm)": np.random.uniform(500, 2000, num_rows),
        "pH Level": np.random.uniform(4.5, 8.5, num_rows),
        "Nutrient Level": np.random.uniform(10, 100, num_rows),
        "Sunlight Exposure (hrs)": np.random.uniform(4, 12, num_rows),
        "Pest Presence": np.random.choice(["Yes", "No"], num_rows),
        "Disease Symptoms": np.random.choice(["Spots", "Discoloration", "Wilting", "None"], num_rows),
        "Leaf Area (cm^2)": np.random.uniform(20, 200, num_rows),
        "Disease Type": np.random.choice(disease_types, num_rows),
        "Yield (kg/acre)": np.random.uniform(500, 3000, num_rows),
        "Disease Severity": np.random.choice(range(1, 11), num_rows),
    }

    df = pd.DataFrame(data)
    # One-hot encoding categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    return df

def train_model(df):
    # Preprocess dataset
    X = df.drop(columns=["Disease Type_None"])  # Target is "Disease Type"
    y = df["Disease Type_None"]
    
    # Train a RandomForestClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X.columns

# App UI
st.title("Plant Disease Prediction")
st.write("This app predicts the type of disease in plants based on various features.")

# Input features for prediction
plant_type = st.selectbox("Select Plant Type", ["Wheat", "Rice", "Tomato", "Corn", "Potato"])
region = st.selectbox("Select Region", ["North", "South", "East", "West"])
soil_type = st.selectbox("Select Soil Type", ["Sandy", "Loamy", "Clay", "Silty"])
temperature = st.slider("Temperature (°C)", 15, 40, 25)
humidity = st.slider("Humidity (%)", 30, 90, 60)
rainfall = st.slider("Rainfall (mm)", 500, 2000, 1000)
ph_level = st.slider("pH Level", 4.5, 8.5, 6.5)
nutrient_level = st.slider("Nutrient Level", 10, 100, 50)
sunlight = st.slider("Sunlight Exposure (hrs)", 4, 12, 8)
pest_presence = st.selectbox("Pest Presence", ["Yes", "No"])
disease_symptoms = st.selectbox("Disease Symptoms", ["Spots", "Discoloration", "Wilting", "None"])
leaf_area = st.slider("Leaf Area (cm²)", 20, 200, 100)
disease_severity = st.slider("Disease Severity", 1, 10, 5)

# Prepare the input data for prediction
input_data = {
    "Plant Type_Potato": int(plant_type == "Potato"),
    "Plant Type_Rice": int(plant_type == "Rice"),
    "Plant Type_Tomato": int(plant_type == "Tomato"),
    "Plant Type_Wheat": int(plant_type == "Wheat"),
    "Region_South": int(region == "South"),
    "Region_West": int(region == "West"),
    "Soil Type_Loamy": int(soil_type == "Loamy"),
    "Soil Type_Sandy": int(soil_type == "Sandy"),
    "Soil Type_Silty": int(soil_type == "Silty"),
    "Pest Presence_Yes": int(pest_presence == "Yes"),
    "Disease Symptoms_Spots": int(disease_symptoms == "Spots"),
    "Disease Symptoms_Wilting": int(disease_symptoms == "Wilting"),
    "Temperature (C)": temperature,
    "Humidity (%)": humidity,
    "Rainfall (mm)": rainfall,
    "pH Level": ph_level,
    "Nutrient Level": nutrient_level,
    "Sunlight Exposure (hrs)": sunlight,
    "Leaf Area (cm^2)": leaf_area,
    "Disease Severity": disease_severity
}

# Button for prediction
if st.button("Predict Disease Type"):
    # Load data and train the model
    df = load_data()
    model, accuracy, feature_columns = train_model(df)
    
    # Make sure the input_data matches the columns from training (including missing columns)
    missing_cols = set(feature_columns) - set(input_data.keys())
    for col in missing_cols:
        input_data[col] = 0  # Set missing columns to 0 (since they weren't provided by the user)
    
    # Ensure the columns are in the correct order
    input_df = pd.DataFrame([input_data])[feature_columns]
    
    # Making prediction
    prediction = model.predict(input_df)
    prediction_prob = model.predict_proba(input_df)
    
    # Displaying results
    st.write(f"Prediction: The plant is infected with **{prediction[0]}** disease")
    st.write(f"Prediction Probability: {prediction_prob[0]}")
    
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
