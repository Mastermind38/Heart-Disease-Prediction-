import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="Heart Doc", page_icon="💓", layout="wide")
st.title("💓 Heart Disease Prediction Model")

# --- 1. LOAD DATA ---
try:
    df = pd.read_csv(r"E:\Data analyze python\Heart_Disease_Prediction.csv")
    
    # Assume the standard 13 columns + Target exist.
    # If your CSV has different names, you might need to rename them in Excel first.
    # Standard names: Age, Sex, CP, BP, Chol, FBS, EKG, MaxHR, ExerciseAngina, ST_Depression, Slope, Vessels, Thalassemia, Target
    
    # SEPARATE FEATURES (X) AND TARGET (y)
    # We drop the 'Heart Disease' column to get all 13 other columns as inputs
    X = df.drop(columns=['Heart Disease'])  # Assuming 'Heart Disease' is the target column name
    y = df['Heart Disease']
    
    # Train the model on ALL data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000) # Increased iterations for better accuracy
    model.fit(X_train, y_train)
    
    # Show Accuracy
    acc = accuracy_score(y_test, model.predict(X_test))
    st.sidebar.success(f"✅ Model Trained on {len(df)} records")
    st.sidebar.write(f"📊 Accuracy: {acc * 100:.2f}%")

except Exception as e:
    st.error(f"⚠️ Error: {e}")
    st.stop()

# --- 2. USER INPUTS (13 Questions) ---
st.write("### Enter Full Patient Details")

# We create 3 columns to make the form look nice
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Personal Info")
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: Typical Angina, 1: Atypical, 2: Non-Anginal, 3: Asymptomatic")
    bp = st.number_input("Resting Blood Pressure", 80, 200, 120)

with col2:
    st.subheader("2. Vitals & Tests")
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
    maxhr = st.number_input("Max Heart Rate", 60, 220, 150)

with col3:
    st.subheader("3. Advanced Heart Checks")
    exang = st.selectbox("Exercise Induced Angina?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST", [1, 2, 3])
    ca = st.selectbox("Major Vessels Colored by Flourosopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3], help="1: Normal, 2: Fixed Defect, 3: Reversable Defect")

# --- 3. PREDICT ---
if st.button("Analyze Full Risk Profile"):
    # Create the list of 13 inputs in the exact order of the CSV
    user_data = [[age, sex, cp, bp, chol, fbs, restecg, maxhr, exang, oldpeak, slope, ca, thal]]
    
    prediction = model.predict(user_data)
    probability = model.predict_proba(user_data)[0][1]

    st.write("---")
    if prediction[0] == 1:
        st.error(f"⚠️ **HIGH RISK DETECTED** (Probability: {probability:.1%})")
        st.write("The model suggests a high likelihood of heart disease based on these 13 factors.")
    else:
        st.success(f"✅ **LOW RISK** (Probability: {probability:.1%})")
        st.write("The patient's heart profile looks healthy.")