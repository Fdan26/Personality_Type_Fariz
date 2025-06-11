# Load model
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

@st.cache_resource
def load_personality_model():
    return load_model('personality_model.h5')

model = load_personality_model()
class_labels = {0: "Introvert", 1: "Extrovert"}

st.title("Personality Prediction App")

# Input fields
time_spent_alone = st.slider("Time Spent Alone (hours per day)", 0, 24, 0)
stage_fear = st.selectbox("Do you have stage fear?", ["Yes", "No"])
social_event_attendance = st.slider("Frequency of attending Social Event (0–10)", 0, 10, 0)
going_outside = st.slider("Frequency of Going Outside (0–10)", 0, 10, 0)
drained_after_socializing = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])
friends_circle_size = st.slider("Number of Close Friends (0–15)", 0, 15, 0)
post_frequency = st.slider("Frequency of posting on social media (0–10)", 0, 10, 0)

# Convert categorical to numerical
stage_fear_val = 1 if stage_fear == "Yes" else 0
drained_val = 1 if drained_after_socializing == "Yes" else 0

# Prepare input sample
input_sample = np.array([[time_spent_alone, stage_fear_val, social_event_attendance,
                          going_outside, drained_val, friends_circle_size, post_frequency]])

# Predict and display result
if st.button("Predict Personality"):
    pred_probs = model.predict(input_sample)
    pred_class = pred_probs.argmax(axis=1)[0]
    st.success(f"Predicted Personality: **{class_labels[pred_class]}**")