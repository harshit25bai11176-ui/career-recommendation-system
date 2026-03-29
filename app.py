import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🎯 Career Recommendation System")

st.write("Enter your skills and marks")

maths = st.slider("Maths", 0, 100)
physics = st.slider("Physics", 0, 100)
cs = st.slider("Computer Science", 0, 100)
communication = st.slider("Communication Skill", 0, 100)
creativity = st.slider("Creativity", 0, 100)

if st.button("Predict Career"):
    data = np.array([[maths, physics, cs, communication, creativity]])
    result = model.predict(data)

    st.success(f"✅ Recommended Career: {result[0]}")