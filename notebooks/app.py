# fixed
import streamlit as st
from scipy.sparse import hstack

model = joblib.load("models/customer_support_intent_model.joblib")
word_vectorizer = joblib.load("models/customer_support_word_vectorizer.joblib")
char_vectorizer = joblib.load("models/customer_support_char_vectorizer.joblib")


def predict_intent(message):
    word_features = word_vectorizer.transform([message])
    char_features = char_vectorizer.transform([message])
    features = hstack([word_features, char_features])
    return model.predict(features)[0]


st.title("Customer Support Intent Predictor")
st.write("Enter a customer message to predict its intent.")

message = st.text_area("Customer message")

if st.button("Predict intent"):
    if message.strip():
        prediction = predict_intent(message)
        st.success(f"Predicted intent: {prediction}")
    else:
        st.warning("Please enter a message.")
