import streamlit as st
import pickle

# -----------------------------
# Load trained model + vectorizer
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# -----------------------------
# Page UI
# -----------------------------
st.title("🎓 Student Feedback Analyzer")

feedback = st.text_area("Enter Feedback")

# -----------------------------
# Prediction Function
# -----------------------------
def analyze_feedback(text):
    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)

    sentiment = label_encoder.inverse_transform(prediction)[0]

    return sentiment

# -----------------------------
# Simple Rule-based extras
# -----------------------------
def get_category(text):
    if "teacher" in text or "class" in text:
        return "Faculty"
    return "General"

def get_issue(text):
    if "example" in text:
        return "Teaching Quality"
    return "General Issue"

def get_suggestion(text):
    if "example" in text:
        return "Provide more practical examples"
    return "No suggestion"

# -----------------------------
# Button
# -----------------------------
if st.button("Analyze Feedback"):
    try:
        sentiment = analyze_feedback(feedback)

        category = get_category(feedback.lower())
        issue = get_issue(feedback.lower())
        suggestion = get_suggestion(feedback.lower())

        st.success("Analysis Complete")

        st.write(f"Category → {category}")
        st.write(f"Sentiment → {sentiment}")
        st.write(f"Issue → {issue}")
        st.write(f"Suggestion → {suggestion}")

    except Exception as e:
        st.error(f"Model Error: {str(e)}")