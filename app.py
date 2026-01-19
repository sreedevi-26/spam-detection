import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Spam Detector", page_icon="📩")

# Load trained model safely
@st.cache_resource
def load_model():
    try:
        with open("spam_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

model = load_model()

st.title("📩 Spam Detection App")
st.write("Enter email/message features to check if it is spam")

# If model not found, show error but keep UI alive
if model is None:
    st.error("❌ Model file not found. Make sure 'spam_model.pkl' is in your GitHub repo.")
    st.stop()

# Input fields
num_links = st.number_input("Number of links", min_value=0, step=1)
num_words = st.number_input("Number of words", min_value=0, step=1)
has_offer = st.selectbox("Contains offer?", [0, 1])
sender_score = st.slider("Sender score", 0.0, 1.0, 0.5)
all_caps = st.selectbox("All caps text?", [0, 1])

# Predict button
if st.button("Predict"):
    try:
        input_data = np.array(
            [[num_links, num_words, has_offer, sender_score, all_caps]],
            dtype=float
        )
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT spam")
    except Exception as e:
        st.error(". Check if model expects the same number of features.")
