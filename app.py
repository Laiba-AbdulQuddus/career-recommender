import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("career_model.pkl")
mlb = joblib.load("skills_encoder.pkl")

# Skills list
all_skills = list(mlb.classes_)

# Degree + Interest options
degree_options = ['BSCS', 'BSIT', 'BS Stats', 'BBA', 'BCom', 'BA Comm', 'BA English', 'BFA', 'MBA']
interest_options = ['Data', 'Design', 'Software', 'Business', 'Finance', 'Marketing', 'Content', 'Web Dev', 'IT', 'Admin', 'Cyber', 'AI']

# Streamlit UI
st.set_page_config(page_title="Career Recommender", layout="centered")
st.title("🎓 Career Recommender System")
st.markdown("Get a career recommendation based on your **skills**, **degree**, **interest**, and **GPA**.")

# Inputs
selected_skills = st.multiselect("🔧 Select Your Skills", options=all_skills)
selected_degree = st.selectbox("🎓 Select Your Degree", degree_options)
selected_interest = st.selectbox("💡 Select Your Area of Interest", interest_options)
gpa = st.slider("📊 Your GPA", min_value=2.0, max_value=4.0, step=0.01)

if st.button("🎯 Recommend Career"):
    # Encode skills
    skill_vector = mlb.transform([selected_skills])
    skills_df = pd.DataFrame(skill_vector, columns=mlb.classes_)

    # Encode degree and interest
    degree_df = pd.get_dummies(pd.Series([selected_degree]), prefix="Degree")
    interest_df = pd.get_dummies(pd.Series([selected_interest]), prefix="Interest")

    # Combine all inputs
    input_df = pd.concat([skills_df, degree_df, interest_df], axis=1)

    # Add GPA
    input_df["GPA"] = gpa

    # Ensure all columns match model input
    model_cols = model.feature_names_in_
    for col in model_cols:
        if col not in input_df.columns:
            input_df[col] = 0  # fill missing columns with 0

    input_df = input_df[model_cols]

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Based on your profile, we recommend: **{prediction}**")
