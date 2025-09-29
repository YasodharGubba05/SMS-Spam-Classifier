import streamlit as st
import pickle

st.title("SMS Spam Classifier")

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_sms(text):
    text_vect = vectorizer.transform([text])
    pred = model.predict(text_vect)[0]
    return "Spam" if pred == 1 else "Not Spam"

sms_text = st.text_area("Enter SMS text:")
if st.button("Classify"):
    result = predict_sms(sms_text)
    color = "red" if result == "Spam" else "green"
    st.markdown(f"<h3 style='color:{color}'>{result}</h3>", unsafe_allow_html=True)