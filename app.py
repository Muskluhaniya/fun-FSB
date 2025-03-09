import streamlit as st
import pandas as pd
import joblib
import json
from streamlit_lottie import st_lottie

# Load trained model
model = joblib.load("model/b_scoring_model.pkl")


# Load Lottie animation
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


love_animation = load_lottie("data/Animation.json")

# App Title & Animation
st.title("💖 Boyfriend Scoring App 💖")
st_lottie(love_animation, speed=1, height=250, key="love")

st.write("Rate your boyfriend based on different factors and get a score!")

# User Inputs
height = st.slider("📏 Height (cm)", min_value=150, max_value=200, value=170)
nature = st.slider("🌿 Nature (Kindness, Calmness)", min_value=1, max_value=10, value=5)
appearance = st.slider("😎 Appearance", min_value=1, max_value=10, value=5)
attitude = st.slider("🗣️ Attitude", min_value=1, max_value=10, value=5)
loyalty = st.slider("💍 Loyalty", min_value=1, max_value=10, value=5)
romance = st.slider("💘 Romance", min_value=1, max_value=10, value=5)
princess_treatment = st.radio("👑 Princess Treatment?", [1, 0], index=1)
intelligence = st.slider("🧠 Intelligence", min_value=1, max_value=10, value=5)
cuddles_affection = st.slider("🤗 Cuddles & Affection", min_value=1, max_value=10, value=5)
random_surprises_gifts = st.slider("🎁 Random Surprises & Gifts", min_value=1, max_value=10, value=5)
good_morning_night_texts = st.radio("📩 Good Morning/Night Texts?", [1, 0], index=1)
patience_understanding = st.slider("🤝 Patience & Understanding", min_value=1, max_value=10, value=5)
hand_holding_pda = st.radio("✋ Hand Holding & PDA?", [1, 0], index=1)
sense_of_humor = st.slider("😂 Sense of Humor", min_value=1, max_value=10, value=5)
emotional_support = st.slider("💙 Emotional Support", min_value=1, max_value=10, value=5)
financial_stability = st.slider("💰 Financial Stability", min_value=1, max_value=10, value=5)
communication_skills = st.slider("🗣️ Communication Skills", min_value=1, max_value=10, value=5)
adventure_level = st.slider("🏕️ Adventure & Fun Level", min_value=1, max_value=10, value=5)
social_skills = st.slider("👥 Social Skills", min_value=1, max_value=10, value=5)
consistency_effort = st.slider("🔥 Consistency & Effort", min_value=1, max_value=10, value=5)
honesty_transparency = st.slider("📖 Honesty & Transparency", min_value=1, max_value=10, value=5)
willingness_to_improve = st.radio("📈 Willingness to Improve?", [1, 0], index=1)

# Predict Score
if st.button("Calculate Score"):
    input_data = pd.DataFrame(
        [[height, nature, appearance, attitude, loyalty, romance, princess_treatment, intelligence, cuddles_affection,
          random_surprises_gifts, good_morning_night_texts, patience_understanding, hand_holding_pda, sense_of_humor,
          emotional_support, financial_stability, communication_skills, adventure_level, social_skills,
          consistency_effort, honesty_transparency, willingness_to_improve]],
        columns=["Height", "Nature", "Appearance", "Attitude", "Loyalty", "Romance", "Princess_Treatment",
                 "Intelligence", "Cuddles_Affection", "Random_Surprises_Gifts", "Good_Morning_Night_Texts",
                 "Patience_Understanding", "Hand_holding_PDA", "Sense_of_Humor", "Emotional_Support",
                 "Financial_Stability", "Communication_Skills", "Adventure_Level", "Social_Skills",
                 "Consistency_Effort", "Honesty_Transparency", "Willingness_to_Improve"]
    )

    score = model.predict(input_data)[0]
    st.success(f"✨ Boyfriend Score: **{round(score, 2)}/10** ✨")

    # Display different GIFs based on score
    if score <= 7:
        gif_path = "data/low_score.gif"
        st.image(gif_path, caption="Hmm... He needs to improve! 😬", use_container_width=True)
    else:
        gif_path = "data/high_score.gif"
        st.image(gif_path, caption="Great! He is a keeper! ❤️", use_container_width=True)
