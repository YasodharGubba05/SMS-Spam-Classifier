import streamlit as st
import pickle

st.title("SMS Spam Classifier")

# Lazy-load model
model = None
def load_model():
    global model
    if model is None:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
    return model

def classify_sms(message):
    m = load_model()
    return m.predict([message])[0]

# User input
sms_text = st.text_area("Enter SMS text here:")
if st.button("Classify"):
    result = classify_sms(sms_text)
    st.write("Prediction:", result)