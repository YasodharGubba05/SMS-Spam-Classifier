import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# --- Preprocessing and NLTK Setup ---

# Download necessary NLTK data only if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)

# Initialize Porter Stemmer
ps = PorterStemmer()


# --- Text Transformation Function ---

def transform_text(text):
    """
    Performs text preprocessing:
    1. Lowercase
    2. Tokenization
    3. Remove special characters
    4. Remove stop words and punctuation
    5. Stemming
    """
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# --- Load Model and Vectorizer ---

try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Deployment Error: 'vectorizer.pkl' or 'model.pkl' not found.")
    st.info("Please ensure both files are present in the root of your GitHub repository.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model files: {e}")
    st.stop()

# --- Streamlit App Interface ---

st.title("📧 SMS Spam Classifier")
st.markdown("Enter a message to check if it's spam or not.")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms:
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize the preprocessed text
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict using the model
        result = model.predict(vector_input)[0]

        # 4. Display the result
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")