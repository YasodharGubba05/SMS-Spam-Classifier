import streamlit as st
import pickle

st.title("SMS Spam Classifier")

# Lazy-load model and vectorizer
model = None
vectorizer = None


def load_model_vectorizer():
    global model, vectorizer
    if model is None or vectorizer is None:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    return model, vectorizer


def predict_sms(text):
    m, v = load_model_vectorizer()
    text_vect = v.transform([text])
    pred = m.predict(text_vect)[0]

    # Convert numerical prediction to label
    if pred == 1:
        return "Spam"
    else:
        return "Not Spam"


sms_text = st.text_area("Enter SMS text:")
if st.button("Classify"):
    result = predict_sms(sms_text)
    st.write("Prediction:", result)