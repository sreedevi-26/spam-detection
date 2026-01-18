import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 Spam Detection App")
st.write("Enter email/message features to check if it is spam")

# Load trained model safely
@st.cache_resource
def load_model():
    with open("spam_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except:
    st.error("❌ Model file not found. Make sure 'spam_model.pkl' is in the same folder.")
    st.stop()

# Input fields
num_links = st.number_input("Number of links", min_value=0, step=1)
num_words = st.number_input("Number of words", min_value=0, step=1)
has_offer = st.selectbox("Contains offer?", [0, 1])
sender_score = st.slider("Sender score", 0.0, 1.0, 0.5)
all_caps = st.selectbox("All caps text?", [0, 1])

# Predict button
if st.button("Predict"):
    input_data = np.array([[num_links, num_words, has_offer, sender_score, all_caps]])
    prediction = spam_model.pkl.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 This message is SPAM")
    else:
        st.success("✅ This message is NOT spam")
